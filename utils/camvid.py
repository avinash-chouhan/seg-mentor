
import numpy as np, os
from collections import OrderedDict

# assuming the legacy few-classes (11 not 31) annotations are used
NUM_CLASSES = 11

train_images = 376
validation_images = 101
test_images = 237

# assuming (only) the TRAIN images are packed into training.tfrecord,
#  and (only) TEST images into validation.tfrecords
TRAIN_DATASET_SIZE = train_images
VAL_DATASET_SIZE = test_images

camvid31_od_class2color = OrderedDict((
                               ('Animal' ,(64,128,64)),
                               ('Archway' ,(192,0,128)),
                               ('Bicyclist', (0,128, 192)),
                               ('Bridge', (0, 128, 64)),
                               ('Building', (128, 0, 0)),
                               ('Car', (64, 0, 128)),
                               ('CartLuggagePram', (64, 0, 192)),
                               ('Child', (192, 128, 64)),
                               ('Column_Pole', (192, 192, 128)),
                               ('Fence', (64, 64, 128)),
                               ('LaneMkgsDriv', (128, 0, 192)),
                               ('LaneMkgsNonDriv', (192, 0, 64)),
                               ('Misc_Text', (128, 128, 64)),
                               ('MotorcycleScooter', (192, 0, 192)),
                               ('OtherMoving', (128, 64, 64)),
                               ('ParkingBlock', (64, 192, 128)),
                               ('Pedestrian', (64, 64, 0)),
                               ('Road', (128, 64, 128)),
                               ('RoadShoulder', (128, 128, 192)),
                               ('Sidewalk', (0, 0, 192)),
                               ('SignSymbol', (192, 128, 128)),
                               ('Sky' ,(128, 128, 128)),
                               ('SUVPickupTruck', (64, 128,192)),
                               ('TrafficCone', (0, 0, 64)),
                               ('TrafficLight', (0, 64, 64)),
                               ('Train', (192, 64, 128)),
                               ('Tree', (128, 128, 0)),
                               ('Truck_Bus', (192, 128, 192)),
                               ('Tunnel', (64, 0, 64)),
                               ('VegetationMisc', (192, 192, 0)),
                               ('Wall', (64, 192, 0)),
                               ('Void', (0, 0, 0))
                             ))

camvid11_od_class2color = OrderedDict((
     ('Sky', (128, 128, 128)),
     ('Building', (128, 0, 0)),
     ('Pole', (192, 192, 128)),
     ('Road',  (128, 64, 128)),
     ('Pavement', (60, 40, 222)),
     ('Tree', (128, 128, 0)),
     ('SignSymbol', (192, 128, 128)),
     ('Fence', (64, 64, 128)),
     ('Car', (64, 0, 128)),
     ('Pedestrian', (64, 64, 0)),
     ('Bicyclist', (0, 128, 192)),
     ('Unlabeled', (0, 0, 0))
))

camvid_lut = {i:c for i,c in enumerate(camvid11_od_class2color.keys())}
camvid_lut[255] = 'Unlabeled'

classes_name_list = camvid11_od_class2color.keys()
colors_list = camvid11_od_class2color.values()

def decode_colored_to_label_map(seg_image, colors=None):
    """
        Convert a segmentation (normally the ground-truth one) represented as 3-channel image,
        to 1-channel label map with values in [0,...,(num_classes-1)]
    """
    colors = colors or colors_list

    label_map = np.zeros_like(seg_image.shape[:2])
    for classind, colour in enumerate(colors):
        # colour_map = np.full((label.shape[0], label.shape[1], label.shape[2]), colour, dtype=int)
        bool_3D = np.equal(seg_image, colour)
        bool_2D = np.all(bool_3D, axis = -1)
        label_map += classind*bool_2D
    return label_map

def find_weights(trainannot = '/data/camvid/trainannot'):
    ''' compute stats for class weighting'''
    from matplotlib import pyplot as plt
    tot_count = np.zeros(12)
    for imn in os.listdir('/data/camvid/trainannot'):
        im1 = plt.imread('/data/camvid/trainannot/' + imn, np.uint8)
        vals, counts = np.unique(im1, return_counts=True)
        tot_count[vals] += counts

camvid11_train_pxl_counts_by_class = \
   np.array([10682767., 14750079.,623349., 20076880., 2845085.,  6166762.,
             743859.,    714595.,  3719877.,405385.,   184967.,  2503995.])

camvid11_class_weights = 1/np.log(camvid11_train_pxl_counts_by_class+1.02)