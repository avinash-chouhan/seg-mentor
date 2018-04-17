# Hailo-segmentation


- [MetaArchitecture](#architecture)
- [Usage](#usage)
- [Results](#results)
- [Discussion](#results)
- [Previous, similar and future work](#previous-and-similar-work)

## Welcome!
**...to segmentation framework by HailoTech ML team!**
<br> *(..standing on shoulders of [Daniil Pakhomov](http://warmspringwinds.github.io/about/), see Credits below)*

We embrace Tensorflow and specifically tf-slim api and classification nets implementation,
<br>and offer a modular code supporting the classic FCN and various enhancements on top of it:
 - Switching the base FE (feature extractor) out of imagenet-pretrained [slim-models selection](https://github.com/tensorflow/models/tree/master/research/slim)
 - Switching to more sophisticated decoding blocks (beyond s16, s8 skip connections and upsample)

The main runner offers command-line params control of architecture along these lines -
 <br> as well as of training and preprocessing hyperparameters.

We use ```tfrecords``` files and the new TF ```Datasets``` api for data feed,
 and contribute some nice train-time monitoring leveraging this api's advanced features;
 the design should hopefully be easy to extend to other datasets (e.g. Cityscapes, COCO, etc.)

We report results of PASCAL-VOC training with the original FCN
based off several feature extractors (VGG16 as in original paper, lightweight "realtime" nets s.a. ResNet18, Mobilenet, etc.).

Now, sure, the state-of-the-art in semantic segmentation took long strides (pun intended) since the FCN,
and there are many nets which are both faster (some claiming "real-time" - but that depends on HW of course),
and with better accuracy.
<br>However, understanding is still lacking, (even as the researchers do take care to perform ablation studies on their brand new nets), e.g.
**
 - *What's the influence of the base Feature Extractor (FE)?
   To which extent the relative classification performance of per-trained (.e.g ImageNet) FE
   is a predictor of the relative accuracy of a segmentation net using it, per same meta-architecture?
   What will be a good predictor? Can one design a FE specifically for downstream segmentation usage?*
 - *Do specific FEs play better with specific training hyperparams, or specific decode-path enhancements?*
 - *How to design simple, effective and efficient decoder blocks for seg' nets using lightweight (e.g. mobilenet) FEs*
 - *Beyond mIoU - how deeper vectorial performance metrics (e.g. per class, per object size) depend on FE or architecture in general.
    What about (..robustness to..) various failure modes? How that could be controlled?*
 - *What about practical issues? how robust are the mIoU numbers to exact freeze point of training, test subset, etc.?

In addition, discussion of practical issues are hard to come by. E.g. how to monitor training, how to estimate the actual variance/"error-bars" around the reported numbers, what failure modes (possibly critical for certain application) are there and how the net can be architected/trained to handle them - or maybe handled at inference time by multiscale testing, etc.

***We hope the repo will be useful for mix&match experimentation that will advance deeper understanding of above and similar issues, and inch the community closer to practical applications of segmentaion to Autonomous Driving and other tasks***

<br>As an example project, we report here on some minimal decode-path enhancements (***FCN+W***) aimed at making FCN based off Lightweight FEs perform on par as the original VGG based, and share some practical tips and theoretical insight on architecture and training - see *Discussion* below

## Usage
0. Steal a PC with GPU and tensorflow-gpu 1.3 installed.
1. Clone side-by-side this repo and the [Hailo fork of tensorflow/models](https://github.com/hailotech/tf-models-hailofork) : 
```
git clone https://github.com/hailotech/hailo-segmentation
git clone https://github.com/hailotech/tf-models-hailofork
```
(to let seg net builder code call into slim FEs implementations in ```models/research/slim/nets```)


1. Edit and run (utils/pascal_voc_augmented_to_records.py)[#hailo-segmentation/utils/pascal_voc_augmented_to_records.py] according to instructions therein.  (to create the train/val Pascal data in tfrecords format)

2. Download some ImageNet-pretrained slim FE checkpoints using links in [tensorflow/models/research/slim](https://github.com/hailotech/tf-models-hailofork/tree/master/research/slim) into  ```/data/models/```
Download missing ones (ResNet18, ..) from TBD (translated from pytorch-vision by benevolent Hailo ML team). 

3. Run ``` cd hailo-segmentaion && mkdir tmp 
CUDA_VISIBLE_DEVICE=0 python fcn_train.py & ```
to start training plain FCN16-VGG16 with default params (using only 1st GPU if your PC got a few). 

1. Monitor your training by :
  ..* Sanity check - ```tail tmp/<today>_vgg16__<#run>/runlog ``` - verify loss goes down and mIoU goes up..:)
     Note the folder under <repo>/tmp for each training run, in which all relevant outputs are saved - a basic log, config used (in **./runargs**), the checkpoint(s) (the bleeding edge and previous saved each few K iteration), events for tensorboard.
  ..*. Tensorboard: - ```cd tboards && bash createlinks && tensorboard --logdir=. &```
     This way you see all past and present runs under <repo>/tmp in tensorboard and you can use the checkboxes to see curves for a single run or several for side-by-side comparison. Check out the noise around the ***test mIoU*** curve, incorporating randomness of both instantaneous checkpoint and 1/4 of test set used for evaluation) as a crude proxy for the typical deviation of the mIoU a.k.a "error-bars" that would be reported in ideal world (w.o. high stakes on publishing a +0.5% improvement framed as a state-of-the-art advance).

2. After a good night's sleep, run ```python fcn_test.py``` (...or you can test in parallel on second GPU if you were lucky to have it on the PC you stole in stage 1.. try a few different checkpoints from the saturated portion of the training, to get another estimate for robustness of results). 

1. Check out the command line options, and train another net(s), e.g.
```
python fcn_train.py --basenet=resnet_v1_18 --batch_size=20 --learnrate=3e-4 --decaylr=True &
```

1. Check out the ```BaseFcnArch``` interface and train you brand net with modified decoding path. 
1. ...
1. [Profit!!!](http://knowyourmeme.com/memes/profit)

## Architecture

### Net Architecture mix&match by OOP

The architecture is implemented as an abstract class,
which should be subclassed for each architecture (original FCN provided as an example);
the subclasses should implement (aka 'override') the decoding blocks as defined in following drawing:

<div align="center">
<img src="https://github.com/hailotech/hailo-segmentation/blob/master/images/FCNgen.png" width="70%" height="70%"><br><br>
</div>

## Results

### Baseline FCN Results

| Net name      | GFLOPS        | Params  | Pascal <br>VOC mIoU  |
| ------------- |:-------------:| -----:  | ---------------: |
| VGG16  |   ...         |  ...    | ..               |
| Inception V1  |   ...         |  ...    | ..               |
| ResNet_v1_18  |   ...         |  ...    | ..               |
| MobileNet V1  |   ...         | .....   | .....            |
| MobileNet V2  | coming        | soon    | (hopefully)      |


### FCN+W results
...Coming soon...

#### Examples
<div align="center">
<img src="https://github.com/hailotech/hailo-segmentation/blob/master/images/ResNet18_Apr02_HorseRider1.png" width="70%" height="70%"><br><br>
</div>

## Discussion
...Coming soon...

## Previous and similar work
Daniil Pakhomov from https://github.com/warmspringwinds/tf-image-segmentation,
was a major inspiration and code contribution; our repo started as a fork of his one;
most was heavily redesigned but ```/utils```  is still mostly his original.

A similar work is [RTSeg](https://github.com/MSiam/TFSegmentation);
<br> they use Cityscapes and differs a bit in approach to modularization.

## TODOs / "Coming"
- Mobilenet_V2
- Dilation options
- implement some basic architectures over the framework:
    - simple (w.o. ASPP and other bells&whistles) DeepLab(3?)
    - LinkNet - original ResNet18 and Mobilenet_V1/2

## References
[0] [Fully Convolutional Networks for Semantic Segmentation', Long&Shelhamer et al., 2016](https://arxiv.org/pdf/1605.06211.pdf)
<br>[1] [Deep Residual Learning for Instrument Segmentation in Robotic Surgery, Pakhomov et al, 2017](https://arxiv.org/abs/1703.08580)
<br>[2] [RTSEG: REAL-TIME SEMANTIC SEGMENTATION COMPARATIVE STUDY, Mar 2018](https://arxiv.org/pdf/1803.02758.pdf)
