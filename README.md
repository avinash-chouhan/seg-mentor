# Hailo-segmentation


- [Architecture](#Net Architecture mix&match guide)
- [Results & discussion](#Baseline FCN Results)
- [Credits & references, future work](#Credits and similar work)

Welcome to segmentation framework by HailoTech ML team!
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

***We hope the repo will be useful for mix&match experimentation that will advance deeper understanding of above and similar issues***

<br>As an example project, we report here on some minimal decode-path enhancements (***FCN+W***) aimed at making FCN based off Lightweight FEs perform on par as the original VGG based, and share some practical tips and theoretical insight on architecture and training - see *Discussion* below

### Net Architecture mix&match guide

The architecture is implemented as an abstract class,
which should be subclassed for each architecture (original FCN provided as an example);
the subclasses should implement (aka 'override') the decoding blocks as defined in following drawing:

<div align="center">
<img src="https://github.com/hailotech/hailo-segmentation/blob/master/images/FCNgen.png" width="70%" height="70%"><br><br>
</div>

## Baseline FCN Results

| Net name      | GFLOPS        | Params  | Pascal <br>VOC mIoU  |
| ------------- |:-------------:| -----:  | ---------------: |
| VGG16  |   ...         |  ...    | ..               |
| Inception V1  |   ...         |  ...    | ..               |
| ResNet_v1_18  |   ...         |  ...    | ..               |
| MobileNet V1  |   ...         | .....   | .....            |
| MobileNet V2  | coming        | soon    | (hopefully)      |


## FCN+W results
...Coming soon...

#### Examples
<div align="center">
<img src="https://github.com/hailotech/hailo-segmentation/blob/master/images/ResNet18_Apr02_HorseRider1.png" width="70%" height="70%"><br><br>
</div>

## Discussion and Insight
...Coming soon...

## Credits and similar work
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
