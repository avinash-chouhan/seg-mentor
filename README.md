# tf-slim-segmentation

- [Contribution](#contribution)
- [MetaArchitecture](#architecture)
- [Usage](#usage)
- [Results & Discussion](#results-and-discussion)
- [Previous, similar and future work](#previous-and-similar-work)

## WELCOME!
to **tf-slim-based flexible semantic segmentation framework**, 
<br>brought to you by the jovial HailoTech ML team. 
<br>(and some code giants on whose shoulders we stand, e.g. [[Pakhomov](http://warmspringwinds.github.io/about/)], see full *[Credits](#previous-and-similar-work)* below)

<div align="center">
<img src="https://github.com/hailotech/hailo-segmentation/blob/master/media/ResNet18_Apr02_HorseRider1.png" width="90%" height="90%"><br><br>
</div>

<div align="center">
<img src="https://github.com/hailotech/hailo-segmentation/blob/master/media/perfectparking.gif" width="90%" height="90%"><br><br>
</div>

We embrace Tensorflow and specifically tf-slim API and classification nets implementation,
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
   To which extent the relative classification performance of pre-trained (.e.g ImageNet) FE
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
    1. Checkpoints for ImageNet-pretrained slim FE checkpoints using links in [tensorflow/models/research/slim](https://github.com/hailotech/tf-models-hailofork/tree/master/research/slim)
        . If needed, get missing ones (ResNet18, ..) from http://dropbox/TBD (translated from pytorch-vision by benevolent Hailo ML team).
    1. The dataset - [Pascal VOC12 trainval](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar)
(images and somea & labels) and [Berekely SBD ](http://www.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/semantic_contours/benchmark.tg)
(+10K seg. annotations for pascal data) .
    1.  Note: scripts below assume ```/data/models``` as the dir with FE checkpoints,
       and ```/data``` as root dir for dataset (under which *VOCdevkit/VOC2012*, *SBD/benchmark_RELEASE* reside).
       <br>If you (or your admin..) don't agree than ```/data/``` is the greatest root dir ever,
       use ```--datapath, --modelspath``` cli arg to inform the scripts where you've put stuff.

1. **Convert** the train/val Pascal data to tfrecords format by running [utils/pascal_voc_augmented_to_records.py](#hailo-segmentation/utils/pascal_voc_augmented_to_records.py)
  , <br>

1. **Prepare** to run by ```cd hailo-segmentaion && mkdir tmp```
<br>The ```tmp``` folder will host "training folders", one per run,
 to be created the training script.
 A "training folder" will contatin all relevant outputs -
      a basic log (```runlog```), config used (```runargs```), the checkpoint(s)
      (the bleeding edge and previous saved each few K iteration), *events* file
      for *tensorboard* visualization.<br>
      **Naming convention** for is ```tmp/<TODAY>_<NETNAME>__<#RUN>``` <br>

1. **Run** ```python fcn_train.py --g 0 & ```
to start training plain FCN32-VGG16 with default params
(works given stuff is in ```data```, see above)
using only 1st GPU if you're filthy rich and got a few of 'em.

1. **Monitor** your training by :
    1. **Sanity checks**: using the folder documenting this training run:<br>
    ```cat tmp/<THIS-RUN-FOLDER>/runargs``` - the config (here simply the cli arguments or their defaults..) used for this training..
     verify that it's what you wanted to have (architecture, hyperparams, etc.)
    <br> ```tail tmp/<THIS-RUN-FOLDER>/runlog``` - verify loss goes down and mIoU goes up..:)
     Note the folder under <repo>/tmp for each training run,

    1. **Tensorboard**: ```cd tboards && bash createlinks && tensorboard --logdir=. &```
     This way you see all past and present runs under <repo>/tmp in tensorboard a
     nd you can use the checkboxes to see curves for a single run or several for side-by-side
     comparison. Check out the noise around the ***test mIoU*** curve, incorporating randomness of both instantaneous checkpoint and 1/4 of test set used for evaluation) as a crude proxy for the typical deviation of the mIoU a.k.a "error-bars" that would be reported in ideal world
      (w.o. the high stakes on framing a +1% improvement as a tectonic shift of SoA:):).

2. **Test** the monster you've grown so far by ```python fcn_test.py --traindir <THIS-RUN-FOLDER> --g 1```
<br> leveraging your second GPU (first  one is busy training..), as you can't wait... <br>
Now seriously, give it a **20-30 hours** of single-GPU training
(use tensorboard to see *test-mIoU* flattening), then test and behold the converged IoU results.
<br> Note: the mIoU of full test may surprise you with ~+5% discrepancy w.r.t the tensorboard plot. see Discussion below.
Don't be shy and kill the process (find pid by ```ps aux | grep fcn_train```)
 if it burns your precious GPU cycles for no further improvements.
    1. If in doubt re convergence (or robusteness in general), run with ```--afteriter X```
to test an intermediate checkpoint after X batches (check out your options by ```ls tmp/<THIS-TRAIN-DIR>```).
    1. Get a feeling for what it means by visualizing results, rerunning with ```--vizstep 1```.
    1. Segment a specific image of your fancy with ```--singleimagepath```


1. **Customize** - check out ```fcn_train.py``` cli options, train other net(s) with modded process, e.g.:
    ```
    python fcn_train.py --basenet=resnet_v1_18 --batch_size=20 --learnrate=3e-4 --decaylr=True &
    ```
1. **Dive** into the code, check out the ```BaseFcnArch``` interface, write your own subclass, and train you own brand new net with modified decoding path.
1. [...](http://knowyourmeme.com/memes/profit)
1. **[Profit!!!](http://knowyourmeme.com/memes/profit)**

## Architecture

### Modular Net Architecture via OOP
DL community is keen on opensourcing net (inc. segmentation) implementations which is awesome;
 unfortunately each one gets its own repo which starts to be frustrating at times,
 since so much of code (and procedures!) is the same (or ain't but should be!) and can be shared.

Within the **Tensorflow** realm -
**[Slim-models](https://github.com/tensorflow/models/tree/master/research/slim)** is a laudable attempt to bring implementations of some feature extractors to the same ground.
Beyond classification, you got the **[Object Detection API](https://github.com/tensorflow/models/tree/master/research/object_detection)**
but it's different in spirit,
 catering to high-level-adapt&deploy users,
 while discouraging tinkering with architecture.
 <br>Here we try to make a small first step towards *segmentation-models*, using the shoulders of **slim** giants as a launchpad.
 
 ----------------
 
So, the architecture is implemented as an abstract class (```BaseFcnArch```),
which should be subclassed for each architecture;
the subclasses should implement (aka 'override') the decoding blocks as defined in following drawing:

<div align="center">
<img src="https://github.com/hailotech/hailo-segmentation/blob/master/images/FCNgen.png" width="70%" height="70%"><br><br>
</div>

Note that if you choose the red script across the decoder blocks, you get the original FCN.
This is what's implemented in the ```FcnArch``` class, provided as the baseline example of the ```BaseFcnArch``` interface.

## Results and Discussion

We report on results of train+test with the following breakup:
- train with all SBD annotations (11K images)
- test with all VO12-val annotations disjoint from SBD (907 images),
  <br>(some call it RV-VOC12 ("restricted validation") while others use this name for other set.)

We trained with Adam, mostly with 15 epochs with initial learning rate, then reducing X10 for another 15, twice
(for a total of 45), but found that second part is marginally useful while 3rd is mostly redundant
(most nets are converged by that time). So most results are after +-35 epochs.
<br>  * *...Note - that claim is of course dependent on initial rate (and batch size, momentum, etc.),
       we use the (..very roughly..) largest that gives good initial convergence.*

All nets trained with similar hyperparams to decent results which were quite robust to small param changes,
<br> - except for VGG FE which was sensitive to learning rate, requiring smaller values of LR,
 and, in FCN16 variant, was very sensitive to the init of skip-connection params, requiring a zero init.
 <br>That is consistent with original FCN paper(s) which describe two-stage training or
 "all-at-once" (like us) but with ad-hoc scaling of the skip-connections as a primitive normalization.
 Relative to them, we here use Adam optimizer which makes stuff more robust, such that zero-init
 (very roughly) simulates a two-stage process of optimize-FCN32 then finesse with skip-connection..

The other FEs also have BatchNorm which seem to add more robustness - to higher LR, random init of skip-conn., etc.
It does (naturally) require a >1 batch size for train - which is fine, it's also good for GPU efficiency.

Input images were scaled so larger side becomes 512 pixels, then padded to 512x512
 <br>* *Note - that's a parameter, you can change that - inc. just for test time;
    unsurprizingly results are best for preprocessing same as train..
    would be interesting to check with test-set of higher resolution though...)*

Some flip and strech augmentations were applied...

### Baseline FCN Results

| Net name      | GOPS @512pxl        | Params (*1e6)  | Pascal <br>mIoU %  |
| ------------- |:-------------:| -----:  | ---------------: |
| VGG16 - FCN16 [ts1](#ts1) (**) |   ~92         |  ~135    | **66.5**               |
| VGG16 - FCN32 [ts1](#ts1) (***) |   ~92        |  ~135    | **65.5**               |
| Inception V1 - FCN16 [ts1](#ts1)  |   8.5        |  6.04   | **63.7**               |
| Inception V1 - FCN32 [ts1](#ts1)  |   8.4        |  6.02    | **62.0**               |
| ResNet_v1_18 - FCN16 [ts1](#ts1)   |   9.6       |  10.91    | **60.4**           |
| ResNet_v1_18 - FCN32 [ts1](#ts1)   |   9.5         |  10.9    | **59.5**           |
| MobileNet V1 - FCN16 [ts1](#ts1)   |   2.9         | 3.12  | **57.6**            |
| MobileNet V1 - FCN32 [ts1](#ts1)   |   2.8         | 3.1   | **55.5**            |
| MobileNet V2  | coming        | soon    | (hopefully)      |

#### ts1
(training schedule 1): <br>
Adam (std.), LR=3e-4, /=10@15,30ep, bs=16, ~40 epochs.

(* *): LR = 1e-4, const.
( * * *): LR(ini.) = 3e-5

##### ts2 
..coming soon..

#### Discussion

So VGG is significantly better than others, but it's impractical for real-time deployments,
 blowing both memory- and computations- (at high resolutions) requirements.

The inceptionv-v1 comes close to VGG - but not surpassing as apparently proven possible on CityScapes (see [RealTime-FCN](http://tuprints.ulb.tu-darmstadt.de/6893/1/20171023_dissertationMariusCordts.pdf#page=115) ).

The FCN16 skip-connection generally gives a +1-2% mIoU improvement, 
 which is non-negligible but smaller than for original FCN as reported in paper (Adam is good for FCN32 health?), 
 and in fact not much larger than the noise (w.r.t to test (sub)set choice, and exact params checkpoint) which we estimate to be ~0.5-1% (see tensorboard plots). So we have to conclude that the skip-connection contribution is **minor** - as long as it's used if used as simple linear addition after classification layer contracting #channels to 21 (classes)... 

The resources needed by additional bilinear interpolations (for upsampling) are negligible, as well as those for the FCN16 skip-connection; so the ops and params are quite similar as for original imagenet classifiers, up to removal of final 1000-classes layer (-) and resolution change (+).
<br>Note however that params&ops don't tell the whole story, and there are HW-architecture-dependent issues.
<br>For example, in dataflow architectures, special resource allocation is needed for buffering the skip connections.
 <br>That's the reason we don't care to train FCN8 variants since returns are negligible w.r.t the costs.

#### Technical issues:
* Monitor vs. final result - we monitor *test-mIoU* estimate during training by running the net on a quarter (1/4) of the validation set; the resulting signal reflects both data(sub)set and param-point variability in its noise, thus giving a kinda-realistic rough estimate of the error bars on the value.
 <br> However, we reused the same computational graph as the train, switching between train/val data feed - leveraging the  *[feedable iterator](https://www.tensorflow.org/programmers_guide/datasets)* of TF ```Datasets``` (probably designed for this exact purpose). This is different from what happens at real inference time, since the BatchNorm/Dropout use training settings (randomly zeroing activations / using this-batch-stats instead of freezed moving-average mean&std, respectively), and gives lower results, with the delta decreasing with batch size. 
 <br>We may fix this in the future but currently we feel that it serves the purpose of monitoring - relative comparison, detect flattening etc. and in fact may be be an opportunity for insights..

#### Stuff we tried and didn't give improvement
Note these are still coded and can be enabled via command-line params.
You're invited to get inspired and retry (possibly adding your ideas on top..)

- Modified schedules that would work better for some FEs but not others - some signs of effect but nothing drastic..

- Differential learning rate - large (X constant factor) for the "new" (decoder) layers vs. the pre-trained (FE/encoder).

- "Heavy" (high-momentum) optimizer, as prescribed in [FCN paper](https://arxiv.org/pdf/1605.06211.pdf).
  We tried to reproduce it by increasing the ```beta1``` parameter of Adam (analogous to SGD's momentum),
  from 0.9 to 0.99, with various concurrent changes to batch size and learning rate.

Note that all params mentioned are involved in how gradients computed with different images and different param points are averaged.
<br> We couldn't hit a low hanging fruit with the few runs we've made -
but that doesn't mean some metric improvement (and insight on the side) couldn't be found with a disciplined parameter scan :)


Contributions are welcome! :)

### FCN + wide-context results
...Coming soon...

#### Discussion 
...Coming soon...

## Previous and similar work
Big chunks of our code are borrowed from Daniil Pakhomov's a little bit dated [tf-image-segmentation](https://github.com/warmspringwinds/tf-image-segmentation]) repo.
especially in the ```/utils```  package.

Beyond those, it was very useful as a headstart; we went far beyond a fork though, 
 since we felt a better design will yield a repo more widely and deeply useful.
We hope the current repo will be useful for development and research projects without serious redesign, 

An up-to-date work similar to ours is [RTSeg](https://github.com/MSiam/TFSegmentation) (see also ref. to paper below);
Their 'SkipNet' architecture is in fact an FCN, 
  so when they marry that to ResNet18 and Mobilenet_V1 it's similar to corresponding subset of our work. 
<br> They however use Cityscapes, which takes them more towards dataset-specific issues, e.g. additional coarse labels incorporation, etc. They also differ a bit in approach to modularization and software design, hence our separate project.

Many nets making progress towards high-performance high-resolution real-time segmentation were published since just an year ago, let's mention a few milestones that feel seminal to us: 

- [RealTime-FCN](http://tuprints.ulb.tu-darmstadt.de/6893/1/20171023_dissertationMariusCordts.pdf#page=115) - created as baseline for CityScapes by its main curator. Inception-V1 based, surpassing VGG, then improving to >70% mIoU with coarse labels and architecture augmentation ("context modules").
   
- [LinkNet](https://codeac29.github.io/projects/linknet/) - >70% mIoU on CityScapes, ResNet-18 based.
- [MobilenetV2 (+stripped DeepLabV3)](https://arxiv.org/pdf/1801.04381.pdf), see Table7 - >70% mIoU on both CityScapes and Pascal with ~2M params, ~3Gops (@512pxl). Seems to be SoA in efficiency right now.. Note however that they also pre-train on COCO, and don't report an ablation study of the contribution of this (much bigger than pascal & cityscapes) dataset.

Implementing these and more in the framework defined here is one of the next steps for this repo...

## Future Work
 ***Contributions Welcome! :)***

- Train on [COCO stuff&things](https://github.com/nightrome/cocostuff), transfer to pascal
- Mobilenet V2 as another FE option
- Dilation as a parameter.. 
- Implement a (few) known architecture(s) over the framework:
    - DeepLab(3?) - barebones w.o. ASPP branches. Test Mobilenets 1/2, ResNet18
    - LinkNet - original ResNet18, then attempt to switch FE?
    - U-net - similarly..
  <br>reproduce published results and start testing and reporting on mix&match effects (e.g. LinkNet + Mobilenet V2).
- Road datasets - Cityscapes, Mapillary Vistas, ApolloScape 
- Incorporate more FEs e.g. ShuffleNet - by finding or creating tf-slim implementation
- Implement more architectures over the framework, upgrade base API if needed for more complex branching 
  (e.g. ASPP, PSP, ICnet, etc.)

## References
Feature extractors:
1. ResNet - ['Deep Residual Learning for Image Recognition', He et. al., 2015](https://arxiv.org/pdf/1512.03385.pdf)
1. Inception - ['Going Deeper with Convolutions', Szegedy et. al., 2014](https://arxiv.org/pdf/1409.4842)
1. ['MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications', Howard et. al., 2017](https://arxiv.org/abs/1704.04861)
1. ['MobileNetV2: Inverted Residuals and Linear Bottlenecks', Sandler et. al., 2018](https://arxiv.org/pdf/1801.04381)

FCN based Semantic Segmentation
1. ['Fully Convolutional Networks for Semantic Segmentation', Long&Shelhamer et al., 2015](https://people.eecs.berkeley.edu/~jonlong/long_shelhamer_fcn.pdf)
1. ['Fully Convolutional Networks for Semantic Segmentation' (B), Long&Shelhamer et al., 2016](https://arxiv.org/pdf/1605.06211.pdf)
1. ['Deep Residual Learning for Instrument Segmentation in Robotic Surgery', Pakhomov et al, 2017](https://arxiv.org/abs/1703.08580)
1. [Understanding Cityscapes, Ph.D. thesis, Marius Cordts, 2017](http://tuprints.ulb.tu-darmstadt.de/6893/1/20171023_dissertationMariusCordts.pdf)
1. ['RTSEG: Real-time semantic segmentation comparative study', Mar 2018](https://arxiv.org/pdf/1803.02758.pdf)
1. [LinkNet: Exploiting Encoder Representations for Efficient Semantic Segmentation, Chaurasia et. al., 2017](https://arxiv.org/abs/1707.03718)

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
