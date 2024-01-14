import torch
import torch.nn.functional as F
from wfdb import processing
import pandas as pd
import numpy as np
from matplotlib.collections import LineCollection
import matplotlib.pyplot as plt
import torchcam

def cal_pred_res(prob):
    test_pred = []
    for i, item in enumerate(prob):
        tmp_label = []
        tmp_label.append(1 - item[0])
        tmp_label.append(item[0] - item[1])
        tmp_label.append(item[1] - item[2])
        tmp_label.append(item[2])
        test_pred.append(tmp_label)
    return test_pred

value = []
weights = []
def forward_hook(module, args, output):
    # value.append(args)
    value.append(output)


def hook_grad(module, grad_input, grad_output):
    # weights.append(grad_input)
    weights.append(grad_output)


# cam_test(model, show_case[i], (show_res[i], target), 'stage_list.4.block_list.0.conv1.conv', True)
def cam_test(
        model, # Model
        data,  # ECG data signal my shape is torch.Size([1, 187])
        clas,  # a tuple with (predict, ground_truth)
        layer, # target layer to compute the graph like 'stage_list.4.block_list.0.conv1.conv'
        label,
        show_overlay=False,
        save=False):# if show the heatmap of signal

    #####################################################
    # register hooks

    data = data.reshape(-1, 1, 187)
    output_len = data.shape[-1]
    target_layer = model.get_submodule(layer)
    forward_handle = target_layer.register_forward_hook(forward_hook)
    backward_handle = target_layer.register_full_backward_hook(hook_grad)
    # clear list
    weights.clear()
    model.zero_grad()
    # forward
    value.clear()
    out = model(torch.tensor(data))
    out = cal_pred_res(out)
    # get the specific loss at the class cls
    # because the masked code can process the difference between predict and ground truth
    # so the clas was transfered as a tuple of size 2
    loss = out[0][int(clas[0])]
    loss.backward(retain_graph=True)
    # calculate the GradCAM actimap at clas[0]
    # GAP at the grad_map then fill the missing dimension
    # weight = weight[(...,) + (None,) * missing_dims]
    weight = weights[0][0].mean(-1)
    missing_dims = value[0].ndim - weight.ndim
    weight = weight[(...,) + (None,) * missing_dims]
    cam = value[0] * weight
    p_acti_map = F.relu(cam.sum(1))

    # gt class actimap
    weights.clear()
    model.zero_grad()
    loss = out[0][int(clas[1])]
    loss.backward()
    weight = weights[0][0].mean(-1)[(...,) + (None,) * missing_dims]
    cam = value[0] * weight
    g_acti_map = F.relu(cam.sum(1))

    forward_handle.remove()
    backward_handle.remove()

    #######################################################
    # build colored ECG
    #######################################################
    if show_overlay:
        p_acti_map = p_acti_map.detach().numpy()
        g_acti_map = g_acti_map.detach().numpy()
        p_new_acti = processing.resample_sig(p_acti_map.flatten(), p_acti_map.shape[-1], output_len)[0]
        g_new_acti = processing.resample_sig(g_acti_map.flatten(), g_acti_map.shape[-1], output_len)[0]

        p_new_acti = pd.Series(p_new_acti)
        g_new_acti = pd.Series(g_new_acti)
        p_new_acti = p_new_acti.interpolate()
        g_new_acti = g_new_acti.interpolate()
        # Convenient for drawing
        x = np.arange(0, output_len)
        y = data.flatten()

        points = np.array([x, y]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)

        fig, axs = plt.subplots(2, 1, sharex=True, sharey=True)

        dydx = p_new_acti
        # Create a continuous norm to map from data points to colors
        norm = plt.Normalize(dydx.min(), dydx.max())
        lc = LineCollection(segments, cmap='viridis', norm=norm)
        # Set the values used for colormapping
        lc.set_array(dydx)
        lc.set_linewidth(2)
        line = axs[0].add_collection(lc)
        fig.colorbar(line, ax=axs[0])

        dydx = g_new_acti
        # Create a continuous norm to map from data points to colors
        norm = plt.Normalize(dydx.min(), dydx.max())
        lc = LineCollection(segments, cmap='viridis', norm=norm)
        # Set the values used for colormapping
        lc.set_array(dydx)
        lc.set_linewidth(2)
        line = axs[1].add_collection(lc)
        fig.colorbar(line, ax=axs[1])

        axs[0].set_xlim(x.min(), x.max())
        axs[0].set_ylim(y.min()-0.3, y.max()+0.3)
        axs[0].set_title('predict class:' + clas[0].__str__())
        axs[1].set_xlim(x.min(), x.max())
        axs[1].set_ylim(y.min() - 0.3, y.max() + 0.3)
        axs[1].set_title('ground truth class:' + clas[1].__str__())
        if(save == True):
            plt.savefig(f'true/{label}.png')
            plt.close(fig)
        else:
            plt.show()
    ################################################################
    return p_acti_map
