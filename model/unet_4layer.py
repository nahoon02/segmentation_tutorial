import torch
import torch.nn as nn
from torch import Tensor

class conv_block(nn.Module):
    """ convolution block

    """
    def __init__(self, ch_in: int, ch_out: int) -> None:
        """
        Args:
             ch_in: block input channel
             ch_out: block output channel
        """
        super(conv_block, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: shape  = (n,c_int,h,w)

        Returns: shape = (n,c_out,h,w)
        """
        x = self.conv(x)
        return x


class UNet_4layer(nn.Module):
    """ UNet with 4 layers
    model input: 2d image

    """
    def __init__(self, in_channels: int = 1, init_features: int = 32) -> None:
        """
        Args:
            in_channels: image input channel
            init_features: first convolution output channel
        """
        super(UNet_4layer, self).__init__()

        features = init_features
        self.encoder1 = conv_block(in_channels, features)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder2 = conv_block(features, features * 2)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder3 = conv_block(features * 2, features * 4)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder4 = conv_block(features * 4, features * 8)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.bottleneck = conv_block(features * 8, features * 16)

        self.upconv4 = nn.ConvTranspose2d(
            features * 16, features * 8, kernel_size=2, stride=2
        )
        self.decoder4 = conv_block((features * 8) * 2, features * 8)
        self.upconv3 = nn.ConvTranspose2d(
            features * 8, features * 4, kernel_size=2, stride=2
        )
        self.decoder3 = conv_block((features * 4) * 2, features * 4)
        self.upconv2 = nn.ConvTranspose2d(
            features * 4, features * 2, kernel_size=2, stride=2
        )
        self.decoder2 = conv_block((features * 2) * 2, features * 2)
        self.upconv1 = nn.ConvTranspose2d(
            features * 2, features, kernel_size=2, stride=2
        )
        self.decoder1 = conv_block(features * 2, features)

        self.conv = nn.Conv2d(
            in_channels=features, out_channels=1, kernel_size=1
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: shape = (n,c,h,w)
        """
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))

        bottleneck = self.bottleneck(self.pool4(enc4))

        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4)
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)
        return torch.sigmoid(self.conv(dec1))



if __name__ == '__main__':

    model = UNet_4layer(init_features=4)

    n, c, h, w = 2, 1, 256, 256

    input = torch.rand((n, c, h, w))
    output = model(input)

    print(f'input shape = {input.shape}')
    print(f'output shape = {output.shape}')








