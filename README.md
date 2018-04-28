# Hailo-segmentation

- [Contribution](#contribution)
- [MetaArchitecture](#architecture)
- [Usage](#usage)
- [Results](#results)
- [Discussion](#results)
- [Previous, similar and future work](#previous-and-similar-work)

## WELCOME!
**to segmentation framework by HailoTech ML team!**
<br> *(..standing on shoulders of some code giants, e.g. [[Pakhomov](http://warmspringwinds.github.io/about/)], see full *[Credits](#previous-and-similar-work)* below)*

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
 - ***What's the influence of the base Feature Extractor (FE)?
   To which extent the relative classification performance of per-trained (.e.g ImageNet) FE
   is a predictor of the relative accuracy of a segmentation net using it, per same meta-architecture?
   What will be a good predictor? Can one design a FE specifically for downstream segmentation usage?***
 - ***Do specific FEs play better with specific training hyperparams, or specific decode-path enhancements?***
 - ***How to design simple, effective and efficient decoder blocks for seg' nets using lightweight (e.g. mobilenet) FEs***
 - ***Beyond mIoU - how deeper vectorial performance metrics (e.g. per class, per object size) depend on FE or architecture in general.
    What about (..robustness to..) various failure modes? How that could be controlled?***
 - ***What about practical issues? how robust are the mIoU numbers to exact freeze point of training, test subset, etc.?***

In addition, discussion of practical issues are hard to come by. E.g. how to monitor training, how to estimate the actual variance/"error-bars" around the reported numbers, what failure modes (possibly critical for certain application) are there and how the net can be architected/trained to handle them - or maybe handled at inference time by multiscale testing, etc.

## Contribution

#### We hope the repo will be useful for mix&match experimentation that may make the following DIFF:
```diff
+ PROMOTE DEEPER UNDERSTANDING OF ABOVE ISSUES

+ PROMOTE ROBUSTNESS OF SEGMENTATION TECH,
+   AND ITS READYNESS TO PRACTICAL APPLICATIONS IN VEHICLES AND OTHER DOMAINS

```

As an example project, we report here on some minimal decode-path enhancements (***FCN+W***) aimed at making FCN based off Lightweight FEs perform on par as the original VGG based, and share some practical tips and theoretical insight on architecture and training - see *Discussion* below

## Usage
1. **Steal** a PC with GPU and tensorflow-gpu 1.3 installed.
1. **Clone** side-by-side this repo and the [Hailo fork of tensorflow/models](https://github.com/hailotech/tf-models-hailofork) :
    ```bash
    git clone https://github.com/hailotech/hailo-segmentation
    git clone https://github.com/hailotech/tf-models-hailofork
    ```
    (to let seg net builder code call into slim FEs implementations in ```models/research/slim/nets```)

1. **Download (&extract..)** :
    1. checkpoints for ImageNet-pretrained slim FE checkpoints using links in [tensorflow/models/research/slim](https://github.com/hailotech/tf-models-hailofork/tree/master/research/slim) into  ```/data/models/```
        <br>...get missing ones (ResNet18, ..) from http://dropbox/TBD (translated from pytorch-vision by benevolent Hailo ML team).
    1. The dataset - [Pascal VOC12 trainval](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar)
(data & labels) and [Berekely SBD ](http://www.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/semantic_contours/benchmark.tg)
(+10K seg. annotations for pascal data)

1. **Convert** the train/val Pascal data to tfrecords format by running [utils/pascal_voc_augmented_to_records.py](#hailo-segmentation/utils/pascal_voc_augmented_to_records.py)
  , <br>(pass the path to where you downloaded the dataset.)

3. Run ``` cd hailo-segmentaion && mkdir tmp 
    python fcn_train.py --g 0 & ```
to start training plain FCN16-VGG16 with default params
  <br>(using only 1st GPU if the PC  you've stolen in point 1 got a few of 'em).

1. Monitor your training by :
    1. Sanity check - ```tail tmp/<THIS-RUN-FOLDER>/runlog ``` - verify loss goes down and mIoU goes up..:)
     Note the folder under <repo>/tmp for each training run,
      in which all relevant outputs are saved -
      a basic log (```runlog```), config used (```runargs```), the checkpoint(s)
      (the bleeding edge and previous saved each few K iteration), events for tensorboard.
      **Naming convention** is ```tmp/<TODAY>_<NETNAME>__<#RUN>```
    1. Tensorboard: - ```cd tboards && bash createlinks && tensorboard --logdir=. &```
     This way you see all past and present runs under <repo>/tmp in tensorboard a
     nd you can use the checkboxes to see curves for a single run or several for side-by-side
     comparison. Check out the noise around the ***test mIoU*** curve, incorporating randomness of both instantaneous checkpoint and 1/4 of test set used for evaluation) as a crude proxy for the typical deviation of the mIoU a.k.a "error-bars" that would be reported in ideal world (w.o. high stakes on publishing a +0.5% improvement framed as a state-of-the-art advance).

2. After a good night's sleep, run ```python fcn_test.py``` (...or you can test in parallel on second GPU if you were lucky to have it on the PC you stole in stage 1.. try a few different checkpoints from the saturated portion of the training, to get another estimate for robustness of results). 

1. Check out the command line options, and train another net(s), e.g.
    ```
    python fcn_train.py --basenet=resnet_v1_18 --batch_size=20 --learnrate=3e-4 --decaylr=True &
    ```
1. Check out the ```BaseFcnArch``` interface, write your own subclass, and train you own brand new net with modified decoding path.
1. [...](http://knowyourmeme.com/memes/profit)
1. [Profit!!!](http://knowyourmeme.com/memes/profit)

## Architecture

### Net Architecture mix&match by OOP

The architecture is implemented as an abstract class (```BaseFcnArch```),
which should be subclassed for each architecture;
the subclasses should implement (aka 'override') the decoding blocks as defined in following drawing:

<div align="center">
<img src="https://github.com/hailotech/hailo-segmentation/blob/master/images/FCNgen.png" width="70%" height="70%"><br><br>
</div>
Note that if you choose the red script across the decoder blocks, you get the original FCN.
This is what's implemented in the ```FcnArch``` class, provided as the baseline example of the ```BaseFcnArch``` interface.

## Results

We report on results of train+test with the following breakup:
- train with all SBD annotations (11K images)
- test with all VO12-val annotations disjoint from SBD (907 images),
  <br>(some call it RV-VOC12 ("restricted validation") while others use this name for other set.)

### Baseline FCN Results

| Net name      | GFLOPS        | Params  | Pascal <br>mIoU %  |
| ------------- |:-------------:| -----:  | ---------------: |
| VGG16 - FCN32 [[^ts1] (**)]|   ...         |  ...    | 65.4               |
| VGG16 - FCN16 [ [^ts2] ] |   ...         |  ...    | ..               |
| Inception V1 - FCN16 [ [ts1] ]  |   ...         |  ...    | 63.7               |
| ResNet_v1_18 - FCN16 [ [ts1] ]   |   ...         |  ...    | 60.4           |
| MobileNet V1 - FCN16 [ [ts1] ]   |   ...         | .....   | 57.6            |
| MobileNet V2  | coming        | soon    | (hopefully)      |

[^ts1]: Adam (std.), LR=3e-4, /=10@15,30ep, bs=16, ~40 epochs.

(**)=LR=1e-4


[^ts2]: -- --



### FCN+W results
...Coming soon...

#### Examples
<div align="center">
<img src="https://github.com/hailotech/hailo-segmentation/blob/master/images/ResNet18_Apr02_HorseRider1.png" width="70%" height="70%"><br><br>
</div>

## Discussion
...Coming soon...

## Previous and similar work
Big chunks of our code are borrowed from Daniil Pakhomov's a little bit dated [tf-image-segmentation](https://github.com/warmspringwinds/tf-image-segmentation]) repo.
especially in the ```/utils```  package.

Beyond that it was very useful as a headstart; we went beyond a fork though, since we feel
We hope this repo will be similarly a useful base for further development and research projects.

An up-to-date work similar to oursis [RTSeg](https://github.com/MSiam/TFSegmentation) (see also ref. to paper below);
<br> they use Cityscapes and differs a bit in approach to modularization.

## "Coming" / "Future Work" / "Contributions Welcome" (TODO choose messaging:)
- Break mIoU by object **sizes** (beyond class breakup), compute weighted (m)iIoU
- Beyond PASCAL - KiTTY, Cityscapes, COCO, Mapillary, ApolloScape
  <br> ..transfer learning pipelines between those..?
- More FEs - Mobilenet_V2, ShuffleNet ?
- Dilation options
- implement some basic architectures over the framework, reproduce published results and report on mix&match effects:
    - DeepLab(3?) - barebones w.o. ASPP and other bells&whistles. Test Mobilenets 1/2, ResNet18
    - LinkNet - original ResNet18, then add Mobilenet_V1/2

## References
1. ['Fully Convolutional Networks for Semantic Segmentation', Long&Shelhamer et al., 2015](https://people.eecs.berkeley.edu/~jonlong/long_shelhamer_fcn.pdf)
1. ['Fully Convolutional Networks for Semantic Segmentation' (), Long&Shelhamer et al., 2016](https://arxiv.org/pdf/1605.06211.pdf)
<br>[1] ['Deep Residual Learning for Instrument Segmentation in Robotic Surgery', Pakhomov et al, 2017](https://arxiv.org/abs/1703.08580)
<br>[2] [RTSEG: REAL-TIME SEMANTIC SEGMENTATION COMPARATIVE STUDY, Mar 2018](https://arxiv.org/pdf/1803.02758.pdf)


## Appendix A: Dataset Rants

If you have an hour, do read ```utils/pascal_voc.py``` whose author should be lauded.
(be it Daniil of http://warmspringwinds.github.io/ or one of his spiritual fathers..)

If you have a minute, do read the rant below, it should summarize the status&spirit of things.

...So, VOC PASCAL is kind of mess, the files train/val/test sets X3 challenges (07', 11', 12'),
  with various intersections between the sets of different years/challenges..
then come additional annotations on original files...

Eventually the widest segmentation-relevant stuff seems to come from Berkely project called SBD:
http://home.bharathh.info/pubs/codes/SBD/download.html
Ignore their own rant re non-intersecting train/test with 5623 images.
Here it's done better, using more for train and a bit less for test.

In any case, have no fear, train/val disjointness is ensured in line 546 of pascal_voc.py
