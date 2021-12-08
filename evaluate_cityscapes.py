import os
from PIL import Image
import numpy as np

import torch
import torch.nn as nn
from torch.utils import data, model_zoo

from model.deeplab_mix import Res_Deeplab
from dataset.cityscapes_dataset import CityscapesDataSet
from configs.test_config import get_arguments

IMG_MEAN = np.array((104.00698793,116.66876762,122.67891434), dtype=np.float32)
palette = [128, 64, 128, 244, 35, 232, 70, 70, 70, 102, 102, 156, 190, 153, 153, 153, 153, 153, 250, 170, 30,
           220, 220, 0, 107, 142, 35, 152, 251, 152, 70, 130, 180, 220, 20, 60, 255, 0, 0, 0, 0, 142, 0, 0, 70,
           0, 60, 100, 0, 80, 100, 0, 0, 230, 119, 11, 32]
zero_pad = 256 * 3 - len(palette)
for i in range(zero_pad):
    palette.append(0)


def colorize_mask(mask):
    new_mask = Image.fromarray(mask.astype(np.uint8)).convert('P')
    new_mask.putpalette(palette)

    return new_mask

def main():
    """Create the model and start the evaluation process."""
    args = get_arguments()

    if not os.path.exists(args.save):
        os.makedirs(args.save)

    model = Res_Deeplab(num_classes=args.num_classes)

    if args.restore_from[:4] == 'http':
        saved_state_dict = model_zoo.load_url(args.restore_from)
    else:
        saved_state_dict = torch.load(args.restore_from)

    model_dict = model.state_dict()
    saved_state_dict = {k: v for k, v in saved_state_dict.items() if k in model_dict}
    model_dict.update(saved_state_dict)
    model.load_state_dict(saved_state_dict)

    device = torch.device("cuda" if not args.cpu else "cpu")
    model = model.to(device)

    model.eval()

    testloader = data.DataLoader(CityscapesDataSet(args.data_dir, args.data_list, crop_size=(1024, 512), mean=IMG_MEAN, scale=False, mirror=False, set=args.set),
                                    batch_size=1, shuffle=False, pin_memory=True)

    interp = nn.Upsample(size=(1024, 2048), mode='bilinear', align_corners=True)

    for index, batch in enumerate(testloader):
        # if index % 100 == 0:
        #     print('%d processd' % index)
        image, _, name = batch
        image = image.to(device)
        with torch.no_grad():
            output2, _ = model(image)
            output = interp(output2).cpu().data[0].numpy()

        output = output.transpose(1,2,0)
        output = np.asarray(np.argmax(output, axis=2), dtype=np.uint8)
        output = Image.fromarray(output)

        name = name[0].split('/')[-1]
        output.save('%s/%s' % (args.save, name))

    with open(args.log, 'a') as f:
        f.write(args.restore_from + '\n')
    os.system('python compute_iou.py --gt_dir '+ args.data_dir +'/gtFine/val/ --pred_dir '+ args.save + ' --log ' +args.log)


if __name__ == '__main__':
    main()

