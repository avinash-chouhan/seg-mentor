import numpy as np
from matplotlib import pyplot as plt

        
def visualize_segmentation_adaptive(predictions, orig_image=None,
                                    clabel2cname=None, od_class2color=None): #, cmap=None):
    """Displays segmentation results using colormap that is adapted
    to a number of classes currently present in an image. Use constant colormap is given,
     (adapted subset therein) otherwise create one.
    
    Parameters
    ----------
    orig_image : to blend with segmentation
    predictions : 2d numpy array (width, height)
        Array with integers representing class predictions
    one of (A) or (B):  (will try first to use (A) so that to have colors...)
    (A) od_class2color : ordered dict.
            mapping class name to color vector, with the indices giving the class labels.
    (B) clabel2cname : dict
            A dict that maps class number to its name like
            {0: 'background', 100: 'airplane'}
    Note that both are sized bigger than num_classes, including also the 'unlabeled'
    """

    unique_classes, _relabeled_image = np.unique(predictions, return_inverse=True)
    rlbl_im = _relabeled_image.reshape(predictions.shape)

    from matplotlib.colors import LinearSegmentedColormap as lsc
    if od_class2color:
        no255 = lambda x: len(od_class2color)-1 if x==255 else x
        rlbl_classes_colors_itm = [od_class2color.items()[no255(c)] for c in unique_classes]
        rlbl_colors =  [[y / 256.0 for y in cl_col[1]] for cl_col in rlbl_classes_colors_itm]
        cmap = lsc.from_list('goo', rlbl_colors, len(rlbl_colors))
        labels_names = [str(orig_lbl)+' '+cl_col[0] for orig_lbl, cl_col in zip(unique_classes, rlbl_classes_colors_itm)]
    elif clabel2cname:
        cmap = plt.get_cmap('Paired', len(clabel2cname))
        no255 = lambda x: len(clabel2cname) - 1 if x == 255 else x
        cmap = lsc.from_list('goo', [cmap.colors[no255(c)][:3] for c in unique_classes] , len(unique_classes))
        labels_names = [str(c)+' '+clabel2cname[c] for c in unique_classes]
    else :
        assert 0

    # set limits .5 outside true range
    mat = plt.imshow(rlbl_im, cmap=cmap, alpha=0.8,
                     vmin=np.min(rlbl_im)-.5, vmax=np.max(rlbl_im)+.5)
    if orig_image is not None:
        plt.imshow(orig_image, alpha=0.2)

    # tell the colorbar to tick at integers
    cax = plt.colorbar(mat, ticks=np.arange(np.min(rlbl_im), np.max(rlbl_im)+1))

    # The names to be printed aside the colorbar
    if labels_names:
        cax.ax.set_yticklabels(labels_names)

    return cmap