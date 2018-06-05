# seg-mentor

- [Contribution](#contribution)
- [MetaArchitecture](#architecture)
- [Usage](#usage)
- [Results & Discussion](#results-and-discussion)
- [Previous, similar and future work](#previous-and-similar-work)

#### Hey amigo!
<div align="center">
<img src="https://github.com/hailotech/seg-mentor/blob/master/media/office.gif" width="60%" height="60%"><br><br>
 </div>

 .           ```authors team (chairs, ppl, monitors..:), segmented with custom Inception_v1+FCN trained on Pascal-VOC.```

## WELCOME!
to **seg-mentor** - a flexible semantic segmentation framework built in tensorflow.
We're making the world a better place by offering a modular sandbox in which you can tinker with semantic segmentation networks.
<br>Seg-mentor is quite self-contained, albeit happily reusing code bits and ideas from open-source contribs, see [prev. work](#previous-and-similar-work)...

We support mix&match across a choice of :

**Feature Extractor (aka Encoder ) architectures**
- [x] ResNet(s).
- [x] Inception_v1
- [x] MobileNet_v1
- [x] ...any [tfslim](https://github.com/tensorflow/models/tree/master/research/slim) model with minimal effort.
- [ ] MobileNet_v2

**Segmentation (aka Decoder) architectures**
- [x] FCN
- [ ] U-Net
- [x] LinkNet
- [ ] Dilation, DeepLab(v1,2,3)


**Dataset**:
- [x] Pascal (VOC12)
- [x] CamVid
- [x] COCO (stuff&things)
- [ ] Cityccapes


We trained a few combinations therein; see [Results](#results-and-discussion) below, 
 and feel free to grab pre-trained models from [Releases](https://github.com/hailotech/seg-mentor/releases)
 and use them to segment stuff:

![picture alt width="70%" height="70%"](https://github.com/hailotech/seg-mentor/blob/defcn/media/CamVidLinkNet1.png)
 .    ```..segmented with ResNet18-LinkNet trained on CamVid ```
 
<div align="center">
<img src="https://github.com/hailotech/seg-mentor/blob/defcn/media/camvid_seg.gif" width="50%" height="50%"><br><br>
 </div>
 
We embrace the Tensorflow framework and specifically the tf-slim API (and associated pre-trained classification nets),
<br>and offer a modular code for semantic segmentation with different meta-architectures. Our goal was to make it simple to:
 - *Choose* the base Feature Extractor (FE) aka Encoder from a selection of [pretrained models](https://github.com/tensorflow/models/tree/master/research/slim)
 - *Tinker* with the segmentation meta-architecture (aka Decoder) building on the FE. Currently we support FCN out-of-the-box + a path to upgrade the decoding blocks in 2-3 lines of code along a simple [abstraction](#architecture)
 - *Share* the datafeed/train/test boilerplate between all variants (vs. taming yet-another repo per net/paper).
 - *Config* as much as possbile of architecture and training procedures thru CLI args, w.o. touching code.

We use ```tfrecords``` files and the new TF ```Datasets``` api for data feeding,
 and offer some nice monitoring leveraging this api's advanced features. 
 :Pre-conversion scripts are provided for supported datasets (see checklist above).
 
## Contribution

#### We hope that the repo will be a strong base for your own cool semantic-segmentation project, e.g. exploring some of the [open questions](#appendix-a--semantic-segmentation-terra-incognita), or deploying a lightweight solution to a practical app.  

We also share some practical training tips and thoughts - see *Discussion* below

## Usage

To all the gals out there with semants to segment,
<br> It's easy to do, just follow these steps:

1. **Steal** a linux PC with GPU (if you plan on training), create a python 2.7 virtualenv, with dependencies inc. [tensorflow](https://www.tensorflow.org/install/install_linux) (v1.2 and up, appropriate to your platform (gpu/cpu, CUDA version)) installed inside. You can use ```requirements``` by e.g. editing the TF line to have right version and running ```pip intstall -r requirements``` after activating venv.
1. **Clone** this repo and the [Hailo fork of tensorflow/models](https://github.com/hailotech/tf-models-hailofork) side-by-side, e.g. :
    ```bash
    git clone https://github.com/hailotech/seg-mentor
    git clone https://github.com/hailotech/tf-models-hailofork
    ```
    (to let net builder code call into slim FEs implementations in ```models/research/slim/nets```)

1. **Play** with segmentation using with our pre-trained models,
by downloading them from [seg-mentor/Releases](https://github.com/hailotech/seg-mentor/releases/tag/v0.5),
and jumping to **Test** below.
 Or, check out the [play-with-me notebook](play-with-me.ipynb) using [*jupyter*](http://jupyter.org/install).

<br> If you want to train a segmentation net:
1. **Download** :
    1. Grab checkpoints for ImageNet-pretrained slim FE(s) using link(s) in [tensorflow/models/research/slim](https://github.com/hailotech/tf-models-hailofork/tree/master/research/slim)
        . If needed, get missing ones (ResNet18, ..) from [seg-mentor/Releases](https://github.com/hailotech/seg-mentor/releases/tag/v0.5) (translated from pytorch-vision by benevolent Hailo ML team).
    1. Grab (&extract) the dataset(s) - e.g.
        1. [Pascal VOC](https://github.com/shelhamer/fcn.berkeleyvision.org/tree/master/data/pascal) : 
        ```
           wget http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar.gz
           wget http://www.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/semantic_contours/benchmark.tgz
        ``` 
        for VOC12 images and [Berkely SBD](http://home.bharathh.info/pubs/codes/SBD/download.html) labels, respectively
        
        1. [CamVid](http://mi.eng.cam.ac.uk/research/projects/VideoRec/CamVid/): 
            ```
               git clone https://github.com/alexgkendall/SegNet-Tutorial
               mv SegNet-Tutorial/CamVid /data/camvid              
            ``` 
        1. [COCO-Stuff(&things)](https://github.com/nightrome/cocostuff#downloads) - 3 top files.
    1.  Note: train/test scripts below assume ```/data/models``` as the dir with FE checkpoints,
       and ```/data``` as root dir for datasets
       (under which ```<dataset_family>/(training/validation).tfrecords``` reside).
       <br>If you (or your admin..) don't agree than ```/data/``` is the greatest root dir ever,
       use ```--datapath, --modelspath``` cli arg to inform the scripts where you've put stuff.

1. **Convert** the dataset to TF records format
  (will appear as ```training.tfrecords, validation.tfrecord``` under the datapath dir)
 by running [utils/tfrecordify.py](utils/tfrecordify.py) with command-line args appropriate
 to the dataset family and paths you use. E.g. I had VOC images in separate folder, already in place and used by unrelated obj-det projects..,
 so did ```python utils/tf_recordify.py --dataset_family=pascal_seg --voc_path=/data/VOCdevkit/VOC2012```)
 <br>Example dir tree (from my server) after downloading extracting and converting camvid and pascal
 (with  to convert):

    ```
         /data/camvid
        ├── test
        ├── testannot
        ├── test.txt
        ├── train
        ├── trainannot
        ├── training.tfrecords
        ├── train.txt
        ├── val
        ├── valannot
        ├── validation.tfrecords
        └── val.txt
        /data/pascal_seg
        ├── benchmark_RELEASE
        │   ├── benchmark_code_RELEASE
        │   ├── BharathICCV2011.pdf
        │   ├── dataset
        │   └── README
        ├── benchmark.tgz
        ├── training.tfrecords
        └── validation.tfrecords
        /data/VOCdevkit
        └── VOC2012
        │   ├── Annotations
        │   ├── ImageSets
        │   ├── JPEGImages
        │   ├── labels
        ├── annotations_cache
        ├── results
        ├── test_2007
        ├── test_2012
        ├── VOC2007
     ```


1. **Prepare** to run by ```cd seg-mentor && mkdir tmp```
<br>The ```tmp``` folder will host "training folders", one per run,
 to be created by ```train.py``` upon start (see next stage).
 A "training folder" will contatin all relevant outputs -
      a basic log (```runlog```), config used (```runargs```), the checkpoint(s)
      (the bleeding edge and previous saved each few K iteration), *events* file
      for *tensorboard* visualization.<br>
      **Naming convention** for the folder name is ```tmp/<TODAY>_<NETNAME>__<#RUN>``` <br>

1. **Launch** ```python train.py --g 0 --basenet inception_v1 --dataset_family=pascal &```
to start training something - here a FCN32-INCEPTION_V1 with default params (assumes default dirpaths!),
using only 1st GPU if you're filthy rich and got a few of 'em (see *Tinker* below for more).
Note:
    1. Output is printed to ```tmp/<this_train_run_folder>/runlog``` rather then to console.
    1. Use ```--extended_arch=LinkNet``` to upgrade from FCN to LinkNet
        (or other modern architectures, coming soon..).
    1. Switch to other datasets with ```--dataset-family```;
    if you converted data to path other then ```/data/<dataset_family>/training.tfrecords```
    use the ```--datapath``` arg to adjust.
    1. Check out ``` --help``` for more options and full usage notes;
       if not enough - the code isn't very convoluted, hopefully (-;).

1. **Monitor** your training by :
    1. **Sanity checks**: using the folder documenting this training run:<br>
    ```cat tmp/<THIS-RUN-FOLDER>/runargs``` - the config (here simply the cli arguments or their defaults..) used for this training..
     verify that it's what you wanted to have (architecture, hyperparams, etc.)
    <br> ```tail tmp/<THIS-RUN-FOLDER>/runlog``` - verify loss goes down and mIoU goes up over time..:)
     Note the folder under <repo>/tmp for each training run,

    1. **Tensorboard**: ```cd tboards && bash createlinks && tensorboard --logdir=. &```
     This way you see all past and present runs under <repo>/tmp in tensorboard 
    and you can use the checkboxes to see curves for a single run or several for side-by-side
     comparison. Check out the noise around the ***test mIoU*** curve (incorporating randomness of both instantaneous checkpoint and 1/4 of test set used for evaluation) as a crude proxy for the typical deviation of the mIoU a.k.a "error-bars" that would be reported in ideal world
      (w.o. the high stakes on framing a +1% improvement as a tectonic shift of SoA).

1. **Test** the monster you've grown so far by ```python test.py --traindir <THIS-RUN-FOLDER> --g 1```
<br> leveraging your second GPU (first  one is busy training..), as you can't wait... <br>
    * Now seriously, give it a **20-30 hours** of training; use tensorboard to see *test-mIoU* flattening; then consider killing the process
 (find pid by ```ps aux | grep train``` )
 that now burns your precious GPU cycles for no further gains),
test and behold the converged IoU results.
<br> Note: the mIoU of full test may surprise you with ~+5% discrepancy
w.r.t the tensorboard plot. see Discussion below.

    * Note that all net&dataset config (and weights of course) is taken from <train_dir>.
 That's our way to enable using the result of any given training run @test-time.
    * Check out [test](test.py) options with ```python test.py --help```
 (or [click](test.py) for code), use them to e.g.:
    1. Check convergence-status/robustness by using ```--afteriter X```
to test an intermediate checkpoint after X batches (check out your options by ```ls tmp/<THIS-TRAIN-DIR>```).
    1. Get a feeling for what it means by visualizing some val images, e.g. ##20-25 by ```--first2viz=20 --last2viz=25```.
    1. Segment a specific image/movie of your fancy with ```--imagepath``` / ```--moviepath```
    1. Play with the pre-(up/down)-scaling with ```--pixels``` (default is what train used; use 0 for no-prescale).
...Or use [play-with-me notebook](play-with-me.ipynb) instead of CLI. See more rants there..

1. **Tinker** - check out [train](train.py) train options with ```python train.py --help```,
play with training other net(s), and training process(es), e.g.:
    ```
    python train.py --basenet=resnet_v1_18 --batch_size=20 --learnrate=3e-4 --decaylr=True &
    ```
    Think of interesting variations to test, convince your boss/supervisor to buy you 1K gpu-hours, run hyperparameter scan, reach cool insights.. ..publish, credit our help:)
1. **Architect** - dive into [arch.py](arch.py) code, check out the ```BaseFcnArch``` interface, write your own subclass, train you own brand new net - with decoding path augmented and modified to your fancy, reach record-breaking mIoU reflecting your unique genius..
1. **Develop** - read ***future work*** below and lend a hand:)
1. [...](http://knowyourmeme.com/memes/profit)
1. [...](http://knowyourmeme.com/memes/profit)
1. **[Profit!!!](http://knowyourmeme.com/memes/profit)**

## Architecture

### Modular Net Architecture via OOP
The DL community is keen on open-sourcing net (inc. segmentation) implementations which is awesome for reproducible research; unfortunately, the reproductive path is normally thru yet-another repo with its own boilerplate and learning curve.

Within the **Tensorflow** realm -
**[Slim-models](https://github.com/tensorflow/models/tree/master/research/slim)** is a laudable attempt to bring implementations of some feature extractors to the same ground.
Beyond classification, you got the **[Object Detection API](https://github.com/tensorflow/models/tree/master/research/object_detection)**
but it's different in spirit,
 catering to high-level-adapt-deploy users,
 while discouraging tinkering with architecture.
 <br>Here we try to make a small first step towards *segmentation-slim-models*, using the shoulders of **slim** giants as a launchpad.
 
 ----------------
 
So, the architecture is implemented as an abstract class (```BaseFcnArch```),
which should be subclassed for each architecture;
the subclasses should implement (aka 'override') the decoding blocks as defined in following drawing:

<div align="center">
<img src="https://github.com/hailotech/seg-mentor/blob/master/media/FCNgen.png" width="85%" height="85%"><br><br>
</div>

Note that if you choose the red script across the decoder blocks, you get the original FCN.
This is what's implemented in the ```FcnArch``` class, provided as the baseline example of the ```BaseFcnArch``` interface.

Also, note the dashed (..dotted) FCN8 (..4) optional skip connections.
<br>(toggled by ```--fcn8, --fcn4``` flags to [train.py](train.py); got ```--fcn16``` too but we treat that as the norm).

Switching feature extractor (FE) is done **without code change among the currently supported FEs** (VGG, ResNet18/50, Inception_V1 aka googlenet, Mobilenet_V1 - can't commit to having this sentence updated, so just check out the dictionary at the top of ```fcn_arch.py``` ). **To add support for another FE** you'll need an incremental change in the dict and similar places (we're sure you can figure it out), AND a modification of the net in the sister repo (fork of slim-models); like [we did it for inception etc.](https://github.com/hailotech/tf-models-hailofork/commit/c3280c1433f8b64bb0ed28acf191d6c4c777210b):
1. Change net func signature from ```logits = net(images, ...)``` to ```logits = net(images, ..., base_only=False, **kwargs)``` to add bare-FE option while preserving compatibility..
1. In the body of the function add :
```
if base_only:
  return net, end_points
```
between feature extracting stage and global-pool/fully-connected classification head. 


## Results and Discussion

We report the results of train+test on several datasets: :

***CamVid***
- train on 376 "train" images
- test on 234 "test" images

***Pascal-VOC***: (see also [Appendix B](#appendix-b--dataset-rants) )
- train with all SBD annotations (~11K images)
- test with all VOC12-val annotations disjoint from SBD (907 images),
  <br>(some call it RV-VOC12 ("restricted validation") while others use this name for other set.)

We trained with Adam. The first 15 epochs with initial learning rate, then reducing X10 for another 15 (twice)
for a total of 45 epochs. We found that the second stage is marginally useful while the 3rd is mostly redundant
(most nets are already converged by that time). So most results are after +-35 epochs.
<br>  * *...Note - that claim is obviously dependent on the initial learnig rate (and batch size, momentum, etc.),
       we use the (..very roughly..) largest that gives good initial convergence.*

All nets trained were with similar hyperparams and the results were robust to small param changes,
<br> - except for VGG FE which was sensitive to learning rate, requiring smaller values of LR. 
 its FCN16 variant was very sensitive to the init of skip-connection params, requiring a zero init.
 <br>That is consistent with original FCN paper(s) which described two-stage training or
 "all-at-once" (like us) but with ad-hoc scaling of the skip-connections as a primitive normalization.
 Relative to them, we use an Adam optimizer which makes stuff more robust, such that zero-init
 (very roughly) simulates a two-stage process of optimized-FCN32 then fine-tune with skip-connection..

The other FEs also use BatchNorm which seem to add more robustness - to higher LR, random init of skip-conn., etc.
We used a batch size of 16 (but in limited testing found that for reasonable values e.g. 8,32 the results are the same)

Input images were scaled so larger side becomes 512 pixels, then padded to 512x512.
 <br>That's configurable separately for test/train; naturally, results are best for same choice @test & @train (that's what went into the table below); interestingly, even better than w.o. any rescale (each image's native resolution) @test.

Some flip and strech augmentations were applied...

### CamVid Results
| Architecture   | CamVid <br>mIoU %  |
| ------------- | ---------------: |
| ResNet18- LinkNet  | **68.0**               |

### VOC Results

| Architecture (FE, decoder variant)  | GOPS (512x512 img)      | Params (*1e6)  | Pascal <br>mIoU %  |
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
...gain for some of the nets... - yet to be found!

#### Discussion

So VGG is significantly better than others, but it's impractical for real-time deployments,
 blowing both memory- and computations- (at high resolutions) requirements.

The inception-v1 comes close to VGG - but not surpassing as proven possible on CityScapes (see [RealTime-FCN](http://tuprints.ulb.tu-darmstadt.de/6893/1/20171023_dissertationMariusCordts.pdf#page=115) ).

The FCN16 skip-connection generally gives a +1-2% mIoU improvement, 
 which is non-negligible but smaller than for original FCN as reported in paper (Adam is good for FCN32 health?), 
 and in fact not much larger than the noise (w.r.t to test (sub)set choice, and exact params checkpoint) which we estimate to be ~0.5-1% (see tensorboard plots). So we have to conclude that the skip-connection contribution is **minor** - if it's only used as simple linear addition after the classification layer contracting #channels to 21 (classes).

The resources needed by additional bilinear interpolations (for upsampling) are negligible, as well as those for the FCN16 skip-connection; so the ops and params are quite similar as for original imagenet classifiers, up to removal of the final classification layer (-) and resolution change (+).
<br>remmeber that params&ops don't tell the whole story, and there are HW-architecture-dependent issues which effect runtime.
 <br>That's the reason we don't care to train FCN8 variants since returns are negligible w.r.t the costs.

#### Technical issues:
* Monitor vs. final result - we monitor a *test-mIoU* estimate during training by running the net on a quarter (1/4) of the validation set; the resulting signal reflects both data(sub)set and param-point variability in its noise, thus giving a kinda-realistic rough estimate of the error bars on the value.
 <br> However, we reused the same computational graph as the train, switching between train/val data feed - leveraging the  *[feedable iterator](https://www.tensorflow.org/programmers_guide/datasets)* of TF ```Datasets``` (probably designed for this exact purpose). This is different from what happens at test time, since the BatchNorm/Dropout layers use.
 <br>We may fix this in the future but currently we feel that it serves the purpose of monitoring - relative comparison, detect flattening etc. and in fact may be be an opportunity for insights..

#### Stuff we tried and didn't work
Note these are still coded and can be enabled via command-line params.
You're invited to get inspired and retry (possibly adding your ideas on top..)

- Modified schedules that would work better for some FEs but not others - some signs of effect but nothing drastic..

- Differential learning rate - large (X constant factor) for the "new" (decoder) layers vs. the pre-trained (FE/encoder).

- "Heavy" (high-momentum) optimizer, as prescribed in [FCN paper](https://arxiv.org/pdf/1605.06211.pdf).
  We tried to reproduce it by increasing the ```beta1``` parameter of Adam (analogous to SGD's momentum),
  from 0.9 to 0.99, with various concurrent changes to batch size and learning rate.

Note that all params mentioned are involved in how gradients are computed.
<br> We couldn't hit a low hanging fruit with the few runs we've made -
but that doesn't mean some improvement of IoU (and insight on the side) couldn't be found with a disciplined parameter scan.

<br><br>Once again - Contributions are welcome! 

## Previous and similar work

Some structure ideas and chunks of the code are borrowed from [Daniil Pakhomov](http://warmspringwinds.github.io/)'s great (if a bit dated now..) repo [tf-image-segmentation](https://github.com/warmspringwinds/tf-image-segmentation).

It was greatly useful as a headstart; we went far beyond a fork though, 
since we felt a better and up-to-date design will yield a repo more widely and deeply useful.
If it does the trick for you, please do credit our contribution :)

An up-to-date work similar to ours is [RTSeg](https://github.com/MSiam/TFSegmentation) (see also ref. to paper below);
Their 'SkipNet' architecture is in fact an FCN (altouhgh some details are not as in the original paper), 
  so when they marry that to ResNet18 and Mobilenet_V1 it's similar to the corresponding subset of our work.
 
 Another work in similar vein is [Semantic-Segmentation-Suite](https://github.com/GeorgeSeif/Semantic-Segmentation-Suite) which is regularly updated with SoTA models. Modularity w.r.t both nets and datasets is sought but only partially delivered - not much of a shared abstraction layer beyond tf-slim, and CamVid as the only dataset demo (others supported if converted to same format but no demo scripts provided..).

## Contacts (maintainers)
Alex Finkelstein ([github](https://github.com/falex-ml)) & Mark Grobman ([github](https://github.com/grobman)) 
<br>[Hailo Technologies](http://www.hailotech.com/) ([github](https://github.com/hailotech))

## (Possible) Future Work
 ***Contributions Welcome! :)***

- Dilation as a parameter.. 
- Incorporate more FEs - both those in slim-models, s.a. Mobilenet_V2, NASnet and others, e.g. ShuffleNet - by bringing/creating a tf-slim implementation into our slim-models fork.
- Train on [COCO stuff&things](https://github.com/nightrome/cocostuff), transfer to pascal.. 
- Road datasets - Cityscapes, Mapillary Vistas, ApolloScape.. check out cross-transfer..
- Implement a few more known architecture(s) over the framework:
    - DeepLab(3?) - barebones w.o. ASPP branches. Test Mobilenets 1/2, ResNet18
    - LinkNet - DONE. add U-net...?
  <br>reproduce published results and start testing and reporting on mix&match effects (e.g. LinkNet + Mobilenet V2).
- Multiple-scale (aka pyramid) testing, robustness exploration.
- Multi-GPU training
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

## Appendix A:  Semantic Segmentation Terra Incognita
The state-of-the-art in semantic segmentation took long strides (pun intended) since the FCN paper,
with more accurate nets.

However, understanding is still lacking, even in the epsilon-vicinity of the baseline FCN.
There's a lot of terra incognita that seems worth exploring before rushing to state-of-the-art nets.
 Playing with a simple-no-thrills decoder can give us clean insight into some fundemantal questions:

 - ***What's the influence of the base Feature Extractor (FE) a.k.a Encoder on the segmentation task?
   <br>To which extent does the performance of a FE on a classfication task correlate
with its performance on a segmentation task? is this meta-architecture (a.k.a Decoder) dependent?
Are there seg.-performance gains to be made by designing FE specifically for usage as seg. encoder
(trading off classification performance)?***
 - ***Regarding the training/fine-tuning of segmentation nets based on pre-trained FEs,
how does the optimal choice of hyper-parameters depend on choice of FE, decoder?***
 - ***How to optimally build a decoder for segmentation when using lightweight (e.g. mobilenet) FEs?***
 - ***What are the failure modes of the net? How do they depend on the FE or the decoder used?
 How can this be quantified (beyond stamp-collecting visual examples and gross-averaged metrics s.a. mIoU)?***
 - ***How the datafeed (resolution, augmentations) and pre-training on other sets influences the failure modes,
 and here too, are there synergistic effects with arhcitecture (FE, decoder)***

We don't have the answers to any of the above but we feel that a good infrastructure for running quick experiments can help. And if it doesn't - well at least we all had fun, right?

<br> And even before open research questions, there are important practical issues,
worked-out examples of which are hard to come by.
<br>  For instance - how to monitor training, how to estimate the actual variance/"error-bars" around the reported numbers, how robust are the mIoU numbers to exact freeze point of training, test subset, etc. These are important for adaptations and deployments to real-life and real-stakes problems (quite possibly more than another +3% in the metric) - especially to the resource-constraint ones which call for lightweight nets anyways.

Most importantly, this kind of investigation can be undertaken with resources normally available for a researcher of a small team - in stark contrast to acheiving the SoA on high-profile targets which is increasingly prohibitive for non-Google/Facebook players (e.g. requiring search in architecture space..).

## Appendix B: Dataset Rants

### VOC-Pascal

This multi-generation dataset is a bit of mess, the files train/val/test sets X3 challenges (07', 11', 12'),
  with various intersections between the sets of different years/challenges..
then come additional annotations on original files...

Eventually the widest ***segmentation-relevant*** stuff seems to come from Berkely project called SBD:
http://home.bharathh.info/pubs/codes/SBD/download.html
Ignore their own rant re non-intersecting train/test with 5623 images.
Here it's done better, using more for train and a bit less for test.

In any case, have no fear, train/val disjointness is ensured in line 546 of pascal_voc.py
