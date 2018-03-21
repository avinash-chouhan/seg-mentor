# hailo-segmentation

This repo includes Hailo example segmentation networks, complete with:
 - architecture definition code
 - training and testing procedure code
 - utilities for easy user-level mix&match control and config - 
    of architecture (e.g. feature extractor a.k.a encoder), hyperparameters, and dataset feed.
    
 We focus on lightweight nets with (pending) support of our HW and toolchain.
 <br>Some of them are created by adopting a published meta-architecture,
 and replacing the encoder use by authors to a more lightweight one;
 in this case we also retain the original to verify reproduction of published results.
  
Our first entry is the classic FCN net; 
implementation is forked from https://github.com/warmspringwinds/tf-image-segmentation
