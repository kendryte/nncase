import numpy as np
import tensorflow as tf
import copy

_default_process = ['mobilenetv1', 'mobilenetv2', 'inceptionv3', 'inceptionv4', 'resnet50_v2']
_vgg_precess = ['vgg16']


def default_preprocess(data, shape):
    img_data = copy.deepcopy(data)
    if(len(img_data.shape) == 2):
        img_data = np.expand_dims(img_data, axis=2)
        img_data = np.concatenate([img_data, img_data, img_data], axis=2)

    # imagenet preprocess
    img = tf.image.central_crop(img_data, central_fraction=0.875)
    img = tf.image.resize(img, [shape[1], shape[2]], method=tf.image.ResizeMethod.BILINEAR)
    img_data = np.asarray(img).copy()

    # tf model class preprocess
    img_data /= 127.5
    img_data -= 1.0

    return img_data


def vgg_preprocess(data, shape):
    img_data = copy.deepcopy(data)
    # imagenet preprocess
    img = tf.image.central_crop(img_data, central_fraction=0.875)

    img = tf.expand_dims(img, 0)
    img = tf.image.resize_bilinear(img, [shape[1], shape[2]], align_corners=False)
    img = tf.squeeze(img, [0])

    # tf model class preprocess
    img = tf.subtract(img, 0.5)
    img = tf.multiply(img, 2.0)

    img_data = np.asarray(img)

    return img_data


def preprocess(model, img_data, shape):

    if(len(img_data.shape) == 2):
        img_data = np.expand_dims(img_data, axis=2)
        img_data = np.concatenate([img_data, img_data, img_data], axis=2)
    if model in _default_process:
        return default_preprocess(img_data, shape)
    elif model in _vgg_precess:
        return vgg_preprocess(img_data, shape[1], shape[2], False, 256, 256, False)
    else:
        # # TODO: add more dataset test model
        # raise "model not in model_zoo"
        return default_preprocess(img_data, shape)
