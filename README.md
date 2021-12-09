# SM-PPM
This is a Pytorch implementation of our paper "Style Mixing and Patchwise Prototypical Matching for One-Shot Unsupervised Domain Adaptive Semantic Segmentation".

AAAI2022 [[arxiv]]()

The paper does not available right now. We are working on the camera ready version.

## Requirements
* python3.7
* pytorch>=1.5.0
* cuda10.2
## Datasets
[GTA5](https://download.visinf.tu-darmstadt.de/data/from_games/)

[Synthia](http://synthia-dataset.net/downloads/)

[Cityscapes](https://www.cityscapes-dataset.com/)

[Dark Zurich](https://www.trace.ethz.ch/publications/2019/GCMA_UIoU/)

## Pretrained Models
1. The source only model for [GTA5](https://www.dropbox.com/s/jmxw7x0e2h2rh35/GTA5_baseline.pth?dl=0) and [Synthia](https://www.dropbox.com/s/dwc9wao0rj7pewz/SYNTHIA_baseline.pth?dl=0) are provided by [AdaptSegNet](https://github.com/wasidennis/AdaptSegNet). 
2. For day-to-night adaptation, please download the model pretrained on Cityscapes [here](https://www.dropbox.com/s/j21pbunecdvpv8z/Cityscapes_baseline.pth?dl=0).

Download these pretrained models and put them into the pretrained_model folder.

## Training Instruction for GTA5->Cityscapes
Modify the data path of the train.py and evaluate_cityscapes.py. 

```python
bash run.sh
```

### Acknowledgment
Part of our code is from [MixStyle](https://github.com/KaiyangZhou/mixstyle-release) and [AdaptSegNet](https://github.com/wasidennis/AdaptSegNet).
We gratefully thank the authors for their great work.
Also thank the authors of [ASM](https://github.com/RoyalVane/ASM) for introducing this one-shot UDA setting.
