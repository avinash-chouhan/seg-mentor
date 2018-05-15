import tensorflow as tf


def get_goodpixel_logits_and_1hot_labels(labels_numeric_3d, logits_4d, num_classes):
    """
        Returns two tensors of size (num_valid_entries, num_classes).
        The function converts annotation batch tensor input of the size
        (batch_size, height, width) into label tensor (batch_size, height,
        width, num_classes) and then selects only valid entries, resulting
        in tensor of the size (num_valid_entries, num_classes). The function
        also returns the tensor with corresponding valid entries in the logits
        tensor. Overall, two tensors of the same sizes are returned and later on
        can be used as an input into tf.softmax_cross_entropy_with_logits() to
        get the cross entropy error for each entry.
    """
    binary_masks_by_class = [tf.equal(labels_numeric_3d, x) for x in range(num_classes)]
    labels_one_hot_4d_f = tf.to_float(tf.stack(binary_masks_by_class, axis=-1))

    # Find unmasked pixels good for evaluation:
    #   (gives a 2D tensor, flat list of index triples - spacial and batch dimensions are lost here)
    valid_pixel_coord_vectors = tf.where(labels_numeric_3d < num_classes)

    # Select a flat list of the values (which are actually 1-hot vectors) for al valid pixels, giving 2D tensor
    goodpixels_labels_one_hot_2d_f = tf.gather_nd(params=labels_one_hot_4d_f, indices=valid_pixel_coord_vectors)
    goodpixels_logits_2d = tf.gather_nd(params=logits_4d, indices=valid_pixel_coord_vectors)
    
    return goodpixels_labels_one_hot_2d_f, goodpixels_logits_2d