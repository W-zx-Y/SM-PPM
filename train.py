import os
import numpy as np
from einops import rearrange

import torch
import torch.nn as nn
from torch.utils import data
import torch.optim as optim
import torch.backends.cudnn as cudnn

from configs.train_config import get_arguments
from dataset.gta5_dataset import GTA5DataSet
from dataset.cityscapes_dataset import CityscapesDataSet
from model.deeplab_mix import Res_Deeplab
from utils import compute_entropy, adjust_learning_rate, similarity
from losses import UncCELoss

IMG_MEAN = np.array((104.00698793, 116.66876762, 122.67891434), dtype=np.float32)

def main():
    args = get_arguments()
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    device = torch.device("cuda" if not args.cpu else "cpu")

    w, h = map(int, args.input_size.split(','))
    input_size = (w, h)

    w, h = map(int, args.input_size_target.split(','))
    input_size_target = (w, h)

    cudnn.enabled = True
    cudnn.benchmark = True

    # Create network
    model = Res_Deeplab(num_classes=args.num_classes)
    saved_state_dict = torch.load(args.restore_from)
    model_dict = model.state_dict()
    saved_state_dict = {k: v for k, v in saved_state_dict.items() if k in model_dict}
    model_dict.update(saved_state_dict)
    model.load_state_dict(saved_state_dict)

    model.train()
    model.to(device)

    if not os.path.exists(args.snapshot_dir):
        os.makedirs(args.snapshot_dir)

    # load source and target data (one target sample)
    trainloader = data.DataLoader(GTA5DataSet(args.data_dir, args.data_list,
                                              max_iters=args.num_steps * args.batch_size, crop_size=input_size,
                                              scale=args.random_scale, mirror=args.random_mirror, mean=IMG_MEAN),
                                  batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    trainloader_iter = enumerate(trainloader)

    targetloader = data.DataLoader(CityscapesDataSet(args.data_dir_target, args.data_list_target,
                                                     max_iters=args.num_steps * args.batch_size,
                                                     crop_size=input_size_target,
                                                     scale=False, mirror=args.random_mirror, mean=IMG_MEAN,
                                                     set=args.set),
                                   batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
                                   pin_memory=True)

    targetloader_iter = enumerate(targetloader)
    _, batch = targetloader_iter.__next__()
    t_images, _, t_name = batch
    t_images = t_images.to(device)

    # optimizer and loss function
    optimizer = optim.SGD(model.optim_parameters(args),
                          lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)
    optimizer.zero_grad()
    seg_loss = UncCELoss(ignore=255).to(device)


    interp = nn.Upsample(size=(input_size[1], input_size[0]), mode='bilinear', align_corners=True)
    interp_fea = nn.Upsample(size=(128, 256), mode='bilinear', align_corners=True)


    # training iterations
    for i_iter in range(args.num_steps_stop+1):
        optimizer.zero_grad()
        adjust_learning_rate(args, optimizer, i_iter)

        # get the target prediction in the evaluation mode
        with torch.no_grad():
            pred_t, t_feas = model(t_images, style=None)
            _, _, h, w = t_feas[2].shape
            p = 32
            ref_patch4 = rearrange(interp_fea(t_feas[4]), 'b c (h p1) (w p2) -> (h w) b c (p1 p2)', p1=p, p2=p)
            prototypes4 = [torch.mean(ref_patch4[i], dim=2) for i in range(ref_patch4.shape[0])]

        _, batch = trainloader_iter.__next__()

        images, labels, _, _ = batch
        images = images.to(device)
        labels = labels.long().to(device)

        pred, s_feas = model(images, style=t_feas)
        pred = interp(pred)
        entropy_s = torch.from_numpy(compute_entropy(pred)).to(device)

        # achieve the similarity using the feature from layer4
        conf = similarity(s_feas[4], prototypes4)
        conf = torch.max(conf, dim=1)[0]
        conf = conf.unsqueeze(1)
        conf = interp(conf)

        loss = seg_loss(pred, labels, conf*(1-entropy_s))
        loss.backward()
        optimizer.step()

        if i_iter % args.save_pred_every == 0 and i_iter != 0:
            torch.save(model.state_dict(), os.path.join(args.snapshot_dir, 'GTA5_' + str(i_iter) + '.pth'))


if __name__ == '__main__':
    main()

