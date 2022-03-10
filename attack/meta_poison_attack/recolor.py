import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_graphics.math.interpolation import trilinear 
import timeit

# Based on https://github.com/wronnyhuang/metapoison/blob/master/recolor.py

@tf.function
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

    @tf.function
    def _recolor(arg):
        # We suppose the pixel values are rescaled to [0, 1]
        img, yref = arg  # img and colorpert shape [ncolorres, ncolorres, ncolorres, 3]
        #img = tfp.math.batch_interp_regular_nd_grid(img, xrefmin, xrefmax, yref, axis=-4)
        img_shape = img.shape
        img = tf.reshape(img, [img_shape[0], -1, 3])
        img = trilinear.interpolate(grid_3d=yref, sampling_points=img)
        img = tf.reshape(img, img_shape)
        return img

    @tf.function
    def read_meshgrid(colorgrid, inter_idx):
        return tf.gather_nd(colorgrid, inter_idx, batch_dims=1)

    @tf.function
    def trilinear_interpolation(imgs, colorgrid, xrefmin, xrefmax):
                # imges: [N, W, H, 3]
        # meshgrid: [N, #ycolors, #ucolors, #vcolors, 3]
        dtype = imgs.dtype
        gridshape = colorgrid.shape.as_list()[1:]
        
        # Calculate interpolate indices
        inter_idx = tf.Variable(imgs, dtype=dtype)
        #inter_mask = tf.zeros_like(imgs, dtype=dtype)
        for i in range(len(xrefmin)):
            inter_idx[..., i].assign(tf.cast((imgs[..., i] - xrefmin[i]) * gridshape[i], dtype=dtype))

        inter_idx_float = inter_idx % 1
        inter_idx = tf.cast(tf.math.floor(inter_idx_float), tf.int32)

        grid_values = []
        coeffs = []
        idx = tf.Variable(inter_idx, dtype=tf.int32, trainable=False)
        for y_i in [0, 1]:
            for u_i in [0, 1]:
                for v_i in [0, 1]:
                    idx.assign(inter_idx)
                    idx[...,0].assign(idx[...,0]+y_i)
                    idx[...,1].assign(idx[...,1]+u_i)
                    idx[...,2].assign(idx[...,2]+v_i)
                    grid_values.append(read_meshgrid(colorgrid, idx))
                    
                    coeff = tf.cast(tf.ones_like(inter_idx_float), dtype=dtype)
                    coeff = coeff * (y_i*(1-coeff)+(1-y_i)*coeff)
                    coeff = coeff * (u_i*(1-coeff)+(1-u_i)*coeff)
                    coeff = coeff * (v_i*(1-coeff)+(1-v_i)*coeff)
                    coeffs.append(tf.cast(coeff, dtype=dtype))
                    del coeff

        outputs = tf.reduce_mean([value * coeff for value, coeff in zip (grid_values, coeffs)], axis=0)
        return outputs

    #@tf.function
    #def func(elements):
    #    return tf.map_fn(_recolor, elements, dtype=inputs.dtype, parallel_iterations=500, name=name)

    yref = grid + colorperts

    inputs = tf.image.rgb_to_yuv(inputs)
    outputs = trilinear_interpolation(inputs, yref, xrefmin, xrefmax)
    outputs = tf.clip_by_value(outputs, xrefmin, xrefmax)

    #outputs = _recolor((inputs, yref))
    #outputs = func((inputs, yref))
    outputs = tf.image.yuv_to_rgb(outputs)
    return outputs, grid


@tf.function
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