import os.path
import numpy as np
from PIL import Image, ImageDraw
import torchvision.transforms as transforms
from typing import Optional, Tuple, List
from torch import Tensor


def _to_color(image_tensor: Tensor):

    if image_tensor.ndim == 3:
        c, h, w = image_tensor.shape
        n = 1
    elif image_tensor.ndim == 4:
        n, c, h, w = image_tensor.shape

    assert n == 1 and c == 1

    # image_buffer shape (h,w)
    image_buffer = image_tensor.squeeze().numpy()
    image_buffer = image_buffer * 255
    image_buffer = image_buffer.astype(np.uint8)
    alpha_buffer = np.full((h, w), 255, dtype=np.uint8)
    # (h, w, 4), 4 = RGBA
    image_buffer = np.repeat(image_buffer[:, :, np.newaxis], 4, axis=2)

    # set alpha channel = 255
    image_buffer[:, :, 3] = 255

    return image_buffer


def _overlay_image(backgroud_buffer: np.array, foreground_buffer: np.array):

    b_img = Image.fromarray(backgroud_buffer)
    f_img = Image.fromarray(foreground_buffer)

    img = Image.blend(b_img, f_img, 0.5)

    buffer = np.array(img)

    return buffer

def _paste_image(backgroud_buffer: np.array, foreground_buffer: np.array, topx:int, topy:int):

    b_img = Image.fromarray(backgroud_buffer)
    f_img = Image.fromarray(foreground_buffer)

    area = (topx, topy)

    b_img.paste(f_img, area)

    buffer = np.array(b_img)

    return buffer


def _overlay_text(image_buffer: np.ndarray, text: str, textxy: Tuple[int, int] = (20, 20),
                  font_color: Tuple[int, int, int] = (255, 165, 0)) -> np.ndarray:
    """ text overlay on image array

    Args:
        image_buffer: image array
        text: text overlay on image
        textxy: text overlay position
        font_color: text color
    """

    assert image_buffer.ndim == 3

    img = Image.fromarray(image_buffer)
    draw = ImageDraw.Draw(img)

    # add alpha value in font_color
    font_color_list = list(font_color)
    font_color_list.append(255)  # alpha value
    font_color = tuple(font_color_list)

    draw.text(textxy, text, fill=font_color)

    modified_img_buffer = np.asarray(img)

    return modified_img_buffer

GAP = 10

def save_image(ct_tensor:Tensor, gt_mask_tensor:Tensor, pred_mask_tensor:Tensor, score:float,
               output_dir:str, filename:str):

    # buffer shape = (h, w, 4), 4 = RGBA
    ct_buffer = _to_color(ct_tensor)
    gt_mask_buffer = _to_color(gt_mask_tensor)
    pred_mask_buffer = _to_color(pred_mask_tensor)

    # change gt_mask_buffer color into blue
    gt_mask_buffer[:, :, 0:2] = 0

    # change pred_mask_buffer color into red
    pred_mask_buffer[:, :, 1:3] = 0

    ct_gtmask_buffer = _overlay_image(ct_buffer, gt_mask_buffer)
    ct_predmask_buffer = _overlay_image(ct_buffer, pred_mask_buffer)

    # remove alpha channel
    ct_gtmask_buffer = ct_gtmask_buffer[:, :, :-1]
    ct_predmask_buffer = ct_predmask_buffer[:, :, :-1]

    # overlay a text on mask buffer
    ct_buffer = _overlay_text(ct_buffer, filename)
    ct_gtmask_buffer = _overlay_text(ct_gtmask_buffer, 'ground truth')
    ct_predmask_buffer = _overlay_text(ct_predmask_buffer, f'prediction, DSC={score:.2f}')

    # merge three buffers into one
    h, w, _ = ct_buffer.shape
    width = h*3 + GAP*2
    height = w
    background_buffer = np.full((height, width), 255, dtype=np.uint8)
    background_buffer = np.repeat(background_buffer[:, :, np.newaxis], 3, axis=2)

    merged_buffer = _paste_image(background_buffer, ct_buffer, topx=0, topy=0)
    merged_buffer = _paste_image(merged_buffer, ct_gtmask_buffer, topx=w+GAP, topy=0)
    merged_buffer = _paste_image(merged_buffer, ct_predmask_buffer, topx=w*2+GAP*2, topy=0)

    if not os.path.isdir(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    output_path = os.path.join(output_dir, f'{filename[:-4]}_result.png')
    Image.fromarray(merged_buffer).save(output_path)


if __name__ == '__main__':

    ct = Image.open("../dataset/dataset_256/ct/10097039_20200724.018.png")
    gt_mask = Image.open("../dataset/dataset_256/mask/10097039_20200724.018.png")
    pred_mask = Image.open("../dataset/dataset_256/mask/10097039_20200724.018.png")

    ct_buffer = np.array(ct)
    gt_mask_buffer = np.array(gt_mask)
    pred_mask_buffer = np.array(pred_mask)

    ct_tensor = transforms.ToTensor()(ct_buffer.copy())
    gt_mask_tensor = transforms.ToTensor()(gt_mask_buffer.copy())
    pred_mask_tensor = transforms.ToTensor()(pred_mask_buffer.copy())

    save_image(ct_tensor, gt_mask_tensor, pred_mask_tensor, 0.1234,
               '../dataset', '10097039_20200724.018.png')






# if __name__ == '__main__':
#
#     ct = Image.open("../dataset/dataset_256/ct/10097039_20200724.018.png").convert('RGBA')
#     mask = Image.open("../dataset/dataset_256/mask/10097039_20200724.018.png").convert('RGBA')
#
#     mask_buffer = np.array(mask)
#     print(f'mask_buffer shape = {mask_buffer.shape}')
#
#     a_mask = mask_buffer[:, :, 3]
#     print(f'a_mask shape = {a_mask.shape}')
#
#     mask_buffer[:, :, 1:3] = 0
#     color_mask_img = Image.fromarray(mask_buffer)
#
#     new_image = Image.blend(ct, color_mask_img, 0.5)
#
#     new_image.save("../dataset/merged.png")
#
#     Image.fromarray(a_mask).save("../dataset/alpha_mask.png")
#
#     ct_buffer = np.array(ct)
#     ct_alpha_buffer = ct_buffer[:, :, 3]
#     Image.fromarray(ct_alpha_buffer).save("../dataset/alpah_ct.png")
