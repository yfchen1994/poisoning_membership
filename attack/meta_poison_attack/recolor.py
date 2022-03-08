import tensorflow as tf
import tensorflow_probability as tfp
import timeit

# Based on https://github.com/wronnyhuang/metapoison/blob/master/recolor.py

def recolor(inputs, colorperts, name=None, grid=None):
    '''
    :param img: input images on which recolor is applied. Shape [B, H, W, 3]. Type is numpy array
    :param gridshape: Shape [nL, nU, nV]. Number of points along each axis in the colorspace grid
    :return: color-transformed image, color perturbation function trainable parameters
    '''
    gridshape = colorperts.shape.as_list()[1:]
    inputshape = inputs.shape.as_list()
    assert len(inputshape) == 4, 'input to recolor must be rank 4'
    assert len(gridshape) == 4, 'gridshape to recolor must be rank 4'

    # define identity color transform. Shape [ncolorres, ncolorres, ncolorres, 3]
    xrefmin = [0., -.5, -.5]  # minimum/maximum values for LUV channels
    xrefmax = [1., .5, .5]
    if grid is None:
        grid = tf.meshgrid(*[tf.linspace(start=start, stop=stop, num=ncolorres) for start, stop, ncolorres in zip(xrefmin, xrefmax, gridshape)], indexing='ij')
        grid = tf.stack(grid, axis=-1)
        grid = tf.cast(grid, inputs.dtype)

    # take a single image and a color perturbation grid and perform the color transformation
    @tf.function
    def _recolor(arg):
        # We suppose the pixel values are rescaled to [0, 1]
        img, colorpert = arg  # img and colorpert shape [ncolorres, ncolorres, ncolorres, 3]
        img = tf.image.rgb_to_yuv(img)
        yref = grid + colorpert
        img = tfp.math.batch_interp_regular_nd_grid(img, xrefmin, xrefmax, yref, axis=-4)
        return img
    
    # apply _recolor to all images in batch
    outputs = tf.map_fn(_recolor, (inputs, colorperts), dtype=inputs.dtype, name=name)
    outputs = tf.image.yuv_to_rgb(outputs)
    return outputs, grid

def timer(func):
    def inner(*args, **kwargs):
        start = timeit.default_timer()
        func(*args, **kwargs)
        stop = timeit.default_timer()
        print("Time: ", stop-start)
    return inner

def smoothloss(colorperts):
    gridshape = colorperts.shape.as_list()[1:]
    '''get the mean norm of the discrete gradients along each direction in colorspace'''
    # the gradients in each direction are calculated as the difference between the neighboring grid points in colorspace divided by their distance which is 1 / ncolorres
    dpert_y = colorperts[:, :-1, :, :, :] - colorperts[:, 1:, :, :, :]
    dpert_u = colorperts[:, :, :-1, :, :] - colorperts[:, :, 1:, :, :]
    dpert_v = colorperts[:, :, :, :-1, :] - colorperts[:, :, :, 1:, :]
    flattened = tf.concat([tf.reshape((d * ncolorres) ** 2, [-1]) for d, ncolorres in zip([dpert_y, dpert_u, dpert_v], gridshape)], axis=0)
    smoothloss = tf.reduce_mean(flattened)
    return smoothloss

