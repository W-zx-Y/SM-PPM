import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import torch
import torch.nn.functional as F
from torch.autograd import Variable

palette = [128, 64, 128, 244, 35, 232, 70, 70, 70, 102, 102, 156, 190, 153, 153, 153, 153, 153, 250, 170, 30,
           220, 220, 0, 107, 142, 35, 152, 251, 152, 70, 130, 180, 220, 20, 60, 255, 0, 0, 0, 0, 142, 0, 0, 70,
           0, 60, 100, 0, 80, 100, 0, 0, 230, 119, 11, 32]
zero_pad = 256 * 3 - len(palette)
for i in range(zero_pad):
    palette.append(0)


def colorize_mask(mask):
    # mask: numpy array of the mask
    new_mask = Image.fromarray(mask.astype(np.uint8)).convert('P')
    new_mask.putpalette(palette)

    return new_mask


def save_segmap(map, save_path, name):
    output = map.cpu().data[0].numpy()

    output = output.transpose(1, 2, 0)
    output = np.asarray(np.argmax(output, axis=2), dtype=np.uint8)

    output_col = colorize_mask(output)
    # output = Image.fromarray(output)
    # name = name[0].split('/')[-1]
    # output.save('%s/%s' % (save_path, name))

    output_col.save('%s/%s_color.png' % (save_path, name.split('.')[0]))


def compute_entropy(pred):
    output_sm = F.softmax(pred, dim=1).cpu().data[0].numpy().transpose(1, 2, 0)
    output_ent = np.sum(-np.multiply(output_sm, np.log2(output_sm + 1e-30)), axis=2,
                        keepdims=False)
    output_ent = output_ent/np.log2(19)
    return output_ent


def entropymap(pred, save_path, name, compute=True, save=True):
    if compute == True:
        pred = compute_entropy(pred)
    # output_sm = F.softmax(pred).cpu().data[0].numpy().transpose(1, 2, 0)
    # output_ent = np.sum(-np.multiply(output_sm, np.log2(output_sm)), axis=2,
    #                     keepdims=False)
    # output_ent = output_ent/np.log2(19)
    if save == True:
        fig = plt.figure()
        plt.axis('off')
        plt.imsave('%s/%s_entropy.png' % (save_path, name.split('.')[0]), pred, cmap=cm.jet)

    # hx = sns.heatmap(output_ent, cbar=False, cbar_ax=False, cmap=cm.jet)
    # sns_plot = sns.pairplot(hx, hue='species', size=2.5)
    # plt.savefig('%s/%s_entropy.png' % (save_path, name.split('.')[0]))
    # hx.save('%s/%s_entropy.png' % (save_path, name.split('.')[0]))
    # grid_image = make_grid(torch.from_numpy(output_ent), 3, normalize=True,
    #                        range=(0, np.log2(19)))
    # plt.imsave(grid_image.permute(1, 2, 0))


def save_confmap(pred, save_path, name):
    output_sm = pred.cpu().data[0][0].numpy()
    fig = plt.figure()
    plt.axis('off')
    plt.imsave('%s/%s_conf.png' % (save_path, name.split('.')[0]), output_sm, cmap=cm.jet)
    # hx = sns.heatmap(output_ent, cbar=False, cbar_ax=False, cmap=cm.jet)
    # sns_plot = sns.pairplot(hx, hue='species', size=2.5)
    # plt.savefig('%s/%s_entropy.png' % (save_path, name.split('.')[0]))
    # hx.save('%s/%s_entropy.png' % (save_path, name.split('.')[0]))
    # grid_image = make_grid(torch.from_numpy(output_ent), 3, normalize=True,
    #                        range=(0, np.log2(19)))
    # plt.imsave(grid_image.permute(1, 2, 0))

def prob_2_entropy(prob):
    """
    convert probabilistic prediction maps to weighted self-information maps
    """
    n, c, h, w = prob.size()
    return -torch.mul(prob, torch.log2(prob + 1e-30)) / np.log2(c)


def fast_hist(a, b, n):
    k = (a >= 0) & (a < n)
    return np.bincount(n * a[k].astype(int) + b[k], minlength=n ** 2).reshape(n, n)


def per_class_iu(hist):
    return np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))


def loss_calc(pred, label, device):
    """
    This function returns cross entropy loss for semantic segmentation
    """
    # out shape batch_size x channels x h x w -> batch_size x channels x h x w
    # label shape h x w x 1 x batch_size  -> batch_size x 1 x h x w
    label = label.long().to(device)
    return cross_entropy_2d(pred, label)


def cross_entropy_2d(predict, target):
    """
    Args:
        predict:(n, c, h, w)
        target:(n, h, w)
    """
    assert not target.requires_grad
    assert predict.dim() == 4
    assert target.dim() == 3
    assert predict.size(0) == target.size(0), f"{predict.size(0)} vs {target.size(0)}"
    assert predict.size(2) == target.size(1), f"{predict.size(2)} vs {target.size(1)}"
    assert predict.size(3) == target.size(2), f"{predict.size(3)} vs {target.size(3)}"
    n, c, h, w = predict.size()
    target_mask = (target >= 0) * (target != 255)
    target = target[target_mask]
    if not target.data.dim():
        return Variable(torch.zeros(1))
    predict = predict.transpose(1, 2).transpose(2, 3).contiguous()
    predict = predict[target_mask.view(n, h, w, 1).repeat(1, 1, 1, c)].view(-1, c)
    loss = F.cross_entropy(predict, target, size_average=True)
    return loss

def entropy_loss(v):
    """
        Entropy loss for probabilistic prediction vectors
        input: batch_size x channels x h x w
        output: batch_size x 1 x h x w
    """
    assert v.dim() == 4
    n, c, h, w = v.size()
    return -torch.sum(torch.mul(v, torch.log2(v + 1e-30))) / (n * h * w * np.log2(c))


def lr_poly(base_lr, iter, max_iter, power):
    return base_lr * ((1 - float(iter) / max_iter) ** (power))


def adjust_learning_rate(args, optimizer, i_iter):
    lr = lr_poly(args.learning_rate, i_iter, args.num_steps, args.power)
    optimizer.param_groups[0]['lr'] = lr
    if len(optimizer.param_groups) > 1:
        optimizer.param_groups[1]['lr'] = lr * 10


def similarity(s_fea, prototypes):
    conf = [F.cosine_similarity(s_fea, prototype[..., None, None]) for prototype in prototypes]
    conf = torch.stack(conf, dim=1)
    return conf


def one_hot(index, classes):
    size = index.size()[:1] + (classes,)
    view = index.size()[:1] + (1,)
    mask = torch.Tensor(size).fill_(0).cuda()
    index = index.view(view)
    ones = 1.
    return mask.scatter_(1, index, ones)

