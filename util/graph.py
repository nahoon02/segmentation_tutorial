import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import os
from typing import Optional, List


def display_graph_v2(plt: matplotlib.pyplot, model_name: str, start_epoch: int,
                  train_loss_list: List,num_epochs: int = 100, save_graph: bool = False,
                  model_filename: Optional[str] = None, save_dir: Optional[str]=None,
                  **kwargs) -> None:
    """ line style info: [url](https://matplotlib.org/2.1.2/api/_as_gen/matplotlib.pyplot.plot.html)
    Args:
        plt:
        model_name: graph title
        start_epoch: the first point in x axes
        train_loss_list:
        num_epochs: total epochs. x axes
        save_graph: True, graph is saved with jpg format
        model_filename: model filename with extension
        save_dir: save directory
        **kwargs:
    """

    maker_style = [':r^', '-bo', '--gv', '-k*', ':mp', ':cs', '-.y+']

    """
        plot train loss
    """
    plt.subplot(211)
    plt.plot(range(start_epoch, len(train_loss_list) + start_epoch), train_loss_list, '-bo', label='train(loss)')
    plt.xlim(start_epoch, num_epochs)
    plt.xticks(np.arange(start_epoch, num_epochs, step=1))
    plt.legend(loc='upper right')
    plt.title(model_name)

    """
       plot valid performance
    """
    plt.subplot(212)
    for i, (key, value_list) in enumerate(kwargs.items()):
        plt.plot(range(start_epoch, len(value_list) + start_epoch), value_list, maker_style[i], label=key)


    # plot 90% accuracy line
    x_len = num_epochs - start_epoch + 1
    list_90 = [0.9 for _ in range(x_len)]
    plt.plot(range(start_epoch, num_epochs+1), list_90, color='gray', linestyle='dashed')
    list_80 = [0.8 for _ in range(x_len)]
    plt.plot(range(start_epoch, num_epochs+1), list_80, color='gray', linestyle='dashed')

    plt.xlim(start_epoch, num_epochs)
    plt.xticks(np.arange(start_epoch, num_epochs, step=1))
    plt.ylim(0, 1.0)
    plt.yticks(np.arange(0, 1.1, step=0.1))
    plt.legend(loc='lower right')

    plt.pause(0.0001)
    plt.show()
    if save_graph == False:
        plt.clf()

    if save_graph:
        fig_filename = model_filename[:-4]
        fig_savepath = os.path.join(save_dir, fig_filename + '.jpg')
        plt.savefig(fig_savepath)
        print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++')
        print('train graph saved --> ', fig_savepath)
        print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++')
        plt.clf()



def move_figure(f, x, y):
    """Move figure's upper left corner to pixel (x, y)
    """
    backend = matplotlib.get_backend()
    if backend == 'TkAgg':
        mgr = f.get_current_fig_manager()
        mgr.window.wm_geometry("+%d+%d" % (x, y))
    elif backend == 'WXAgg':
        f.canvas.manager.window.SetPosition((x, y))
    else:
        # This works for QT and GTK
        # You can also use window.setGeometry
        f.canvas.manager.window.move(x, y)


if __name__ == '__main__':
    import time

    plt.ion()
    plt.figure(figsize=(7, 9)) # figure size = (width, height)
    #move_figure(plt, 0, 0)

    train_loss_list = [36.0, 36.0]
    valid_accuracy_list = [0.78, 36.0]
    valid_dsc_list = [0.3, 0.5]
    valid_f1_list = [0.5, 0.1]
    display_graph_v2(plt, 'abc', 1, train_loss_list,
                  num_epochs=100, save_graph=True,
                  model_filename='a.pth', save_dir='/home/nahoon/temp',
                  accuracy=valid_accuracy_list, f1=valid_f1_list, dsc=valid_dsc_list)
    time.sleep(1)

