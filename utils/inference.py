import tensorflow as tf

def adapt_network_for_any_size_input(network_definition, multiple):
    """Returns an updated function for network definition that supports input of any size.
    The function creates a new function that is an original function wrapped with
    a special function that resizes the input image to the closest multiple of 'multiple'
    parameter, provides this image into the network, gets output from the network,
    and resizes the prediction to the size of original input image. The updated function
    returns final predictions and not upsampled logits. The approach was inspired by
    matconvnet-fcn library approach.
    
    Parameters
    ----------
    network_definition : function
        A function with original network definition
    multiple : int
        A number representing the multiple of which the input
        image should be of for the specified network. For example,
        for FCN-32s it is 32.
        
    Returns
    -------
    new_network_definition : function
        Updated function representing networks definition.
    """
    
    def new_network_definition(*args, **kwargs):
        
        # The first argument of the network definition
        # should be 'image_batch_tensor'
        if 'image_batch_tensor' in kwargs:

            image_batch_tensor = kwargs['image_batch_tensor']
        else:
            
            image_batch_tensor = args[0]
            args = args[1:]
        
        input_image_shape = tf.shape(image_batch_tensor)

        image_height_width = input_image_shape[1:3]
        image_height_width_float = tf.to_float(image_height_width)

        image_height_width_multiple = tf.round(image_height_width_float / multiple) * multiple
        image_height_width_multiple = tf.to_int32(image_height_width_multiple)

        kwargs['image_batch_tensor'] = tf.image.resize_images(image_batch_tensor, image_height_width_multiple)

        # Perform the wrapped operation
        upsampled_logits_batch = network_definition(*args, **kwargs)

        original_size_logits = tf.image.resize_nearest_neighbor(images=upsampled_logits_batch, size=image_height_width)
        original_size_predictions = tf.argmax(original_size_logits, dimension=3)
        original_size_predictions = tf.expand_dims(original_size_predictions, 3)

        # # TODO: check if it works with logits, maybe there is no need to do argmax
        # pred = tf.argmax(upsampled_logits_batch, dimension=3)
        # temp_pred = tf.expand_dims(pred, 3)
        # original_size_predictions = tf.image.resize_nearest_neighbor(images=temp_pred, size=image_height_width)

        return original_size_predictions
    
    return new_network_definition
