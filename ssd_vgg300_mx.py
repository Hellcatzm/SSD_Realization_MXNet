import mxnet as mx
import numpy as np
from mxnet.gluon import nn
from collections import namedtuple
from util_mx import *

SSDParams = namedtuple('SSDParameters', ['img_shape',
                                         'num_classes',
                                         'feat_layers',
                                         'feat_shapes',
                                         'anchor_sizes',
                                         'anchor_ratios',
                                         'anchor_offset',
                                         'normalizations',
                                         'prior_scaling'
                                         ])


image_shape = (300, 300)
num_classes = 20
feat_layers = [4, 7, 8, 9, 10, 11]
feat_shapes = [(38, 38), (19, 19), (10, 10), (5, 5), (3, 3), (1, 1)]
anchor_sizes = [(21., 45.),
                (45., 99.),
                (99., 153.),
                (153., 207.),
                (207., 261.),
                (261., 315.)]
anchor_ratios = [[2, .5],
                 [2, .5, 3, 1. / 3],
                 [2, .5, 3, 1. / 3],
                 [2, .5, 3, 1. / 3],
                 [2, .5],
                 [2, .5]]
anchor_offset = 0.5
normalizations = [1, 0, 0, 0, 0, 0]


def ssd_model():

    dropout_keep_prob = 0.5
    conv_arch = ((2, 64), (2, 128),
                 (3, 256), (3, 512),
                 (3, 512))
    
    body = nn.Sequential()
    body.add(repeat(*conv_arch[0], pool=False))
    [body.add(repeat(*conv_arch[i])) for i in range(1, len(conv_arch))]

    body.add(nn.Conv2D(channels=1024, kernel_size=3, padding=1, activation='relu'))

    block = nn.Sequential()
    block.add(nn.Dropout(rate=dropout_keep_prob),
              nn.Conv2D(channels=1024, kernel_size=1, padding=0, activation='relu'))
    body.add(block)

    block = nn.Sequential()
    block.add(nn.Dropout(rate=dropout_keep_prob),
              nn.Conv2D(channels=256, kernel_size=1, padding=0, activation='relu'),
              nn.Conv2D(channels=512, kernel_size=3, strides=2, padding=1, activation='relu'))
    body.add(block)

    block = nn.Sequential()
    block.add(nn.Conv2D(channels=256, kernel_size=1, padding=0, activation='relu'),
              nn.Conv2D(channels=512, kernel_size=3, strides=2, padding=1, activation='relu'))
    body.add(block)

    block = nn.Sequential()
    block.add(nn.Conv2D(channels=128, kernel_size=1, padding=0, activation='relu'),
              nn.Conv2D(channels=256, kernel_size=3,  strides=2, padding=1, activation='relu'))
    body.add(block)

    block = nn.Sequential()
    block.add(nn.Conv2D(channels=128, kernel_size=1, padding=0, activation='relu'),
              nn.Conv2D(channels=256, kernel_size=3, padding=0, activation='relu'))
    body.add(block)

    block = nn.Sequential()
    block.add(nn.Conv2D(channels=128, kernel_size=1, padding=1, activation='relu'),
              nn.Conv2D(channels=256, kernel_size=3, padding=0, activation='relu'))
    body.add(block)

    class_predictors = nn.Sequential()
    box_predictors = nn.Sequential()
    l2_normalizes = nn.Sequential()
    for i in range(len(feat_layers)):
        if normalizations[i]:
            l2_normalizes.add(L2_normalize(512))
        num_anchors = len(anchor_sizes[i]) + len(anchor_ratios[i])
        class_predictors.add(nn.Conv2D(num_anchors * 4, 3, padding=1))
        box_predictors.add(nn.Conv2D(num_anchors * (num_classes + 1), 3, padding=1))
    return body, class_predictors, box_predictors, l2_normalizes


def ssd_forward(feat, ssd_mod):
    body, class_predictors, box_predictors, l2_normalizes = ssd_mod
    anchors, class_preds, box_preds = ([] for _ in range(3))
    feat_layers.insert(0, 0)
    # i为feat_layer索引，j为layer具体层数，feat_layers[i]为特征层
    for i in range(1, len(feat_layers)):
        for j in range(feat_layers[i-1], feat_layers[i]):
            feat = body[j](feat)
            print(feat.shape)
        print(feat_layers[i], feat.shape)
        if normalizations[i - 1]:
            feat = l2_normalizes[i-1](feat)
        class_preds.append(class_predictors[i-1](feat))
        box_preds.append(box_predictors[i-1](feat))

        # feat_layers索引从1开始，但是其他特征相关索引从0开始
        layer_anchors = one_layer_anchers(feat, i-1)
        anchors.append(layer_anchors)


def one_layer_anchers(feat, i):
        feat_shape = feat.shape[2:]
        img_shape = np.array(image_shape)
        scale = (img_shape//feat_shape)/img_shape
        y, x = np.mgrid[0:feat_shape[0], 0:feat_shape[1]].astype(np.float32)
        y = (y + anchor_offset) * scale[0]
        x = (x + anchor_offset) * scale[1]

        y = np.expand_dims(y, axis=-1)
        x = np.expand_dims(x, axis=-1)

        num_anchors = len(anchor_sizes[i]) + len(anchor_ratios[i])
        h = np.zeros((num_anchors,), dtype=float)
        w = np.zeros((num_anchors,), dtype=float)

        # Add first anchor boxes with ratio=1.
        h[0] = anchor_sizes[i][0] / img_shape[0]
        w[0] = anchor_sizes[i][0] / img_shape[1]
        di = 1
        if len(anchor_sizes[i]) > 1:
            h[1] = np.sqrt(anchor_sizes[i][0] * anchor_sizes[i][1]) / img_shape[0]
            w[1] = np.sqrt(anchor_sizes[i][0] * anchor_sizes[i][1]) / img_shape[1]
            di += 1
        for i, r in enumerate(anchor_ratios[i]):
            h[i + di] = anchor_sizes[i][0] / img_shape[0] / np.sqrt(r)
            w[i + di] = anchor_sizes[i][0] / img_shape[1] * np.sqrt(r)

        # import matplotlib.pyplot as plt
        # plt.scatter(y, x, c='r', marker='.')
        # plt.grid(True)
        # plt.show()
        # print(h, w)

        return y, x, h, w


if __name__ == '__main__':
    X = mx.ndarray.random.uniform(shape=(1, 1, 304, 304))
    ssd = ssd_model()
    for model in ssd:
        model.initialize()
    ssd_forward(X, ssd)

