import inference
import augmentation
import training
import upsampling
import visualization
import tf_records
import pascal_voc

'''
    Credits:
    Vast majority of this package should be credited to Daniil of http://warmspringwinds.github.io/,
     (or to one of his own spiritual fathers). A healthy round of applaud is in place, really.
     
    We at Hailo added some adaptaions:
        - updated tf_records.py to work with TF Datasets (the modern way of feeding data in tensorflow)
        - somewhat improved visualization
        - records creation as script vs. notebook, with some doc
'''