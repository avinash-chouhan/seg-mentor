import inference
import augmentation
import training
import upsampling
import visualization
import tf_records
import pascal_voc
import camvid

'''
    Package originated from Daniil of http://warmspringwinds.github.io/ (and his predecessors), 
     
    We at Hailo improved on the following parts:
        - updated tf_records.py to work with TF Datasets (the modern way of feeding data in tensorflow)
        - significanly improved visualization
        - records creation as script vs. notebook, with some doc
        - Added recordization for more datasets - COCO(stuff&things), CamVid, ...?
'''
