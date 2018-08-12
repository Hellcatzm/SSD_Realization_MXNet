import sys
import math
import numpy as np
import tensorflow as tf
from collections import namedtuple

from util_tf import tensor_shape

slim = tf.contrib.slim

SSDParams = namedtuple('SSDParameters', ['img_shape',
                                         'num_classes',
                                         'no_annotation_label',
                                         'feat_layers',
                                         'feat_shapes',
                                         'anchor_size_bounds',
                                         'anchor_sizes',
                                         'anchor_ratios',
                                         'anchor_steps',
                                         'anchor_offset',
                                         'normalizations',
                                         'prior_scaling'
                                         ])


class SSDNet(object):
    default_params = SSDParams(
        img_shape=(300, 300),
        num_classes=21,
        no_annotation_label=21,
        feat_layers=['block4', 'block7', 'block8', 'block9', 'block10', 'block11'],
        feat_shapes=[(38, 38), (19, 19), (10, 10), (5, 5), (3, 3), (1, 1)],
        anchor_size_bounds=[0.15, 0.90],
        anchor_sizes=[(21., 45.),
                      (45., 99.),
                      (99., 153.),
                      (153., 207.),
                      (207., 261.),
                      (261., 315.)],
        anchor_ratios=[[2, .5],
                       [2, .5, 3, 1. / 3],
                       [2, .5, 3, 1. / 3],
                       [2, .5, 3, 1. / 3],
                       [2, .5],
                       [2, .5]],
        anchor_steps=[8, 16, 32, 64, 100, 300],
        anchor_offset=0.5,
        normalizations=[1, -1, -1, -1, -1, -1],  # 控制SSD层处理时是否预先沿着HW正则化
        prior_scaling=[0.1, 0.1, 0.2, 0.2]
    )

    def __init__(self, params=None):
        if isinstance(params, SSDParams):
            self.params = params
        else:
            self.params = SSDNet.default_params

    def net(self, input_data, update_feat_shapes=True):
        with slim.arg_scope(self._ssd_arg_scope()):
            output = self._ssd_net(input_data)
            # Update feature shapes (try at least!)
        if update_feat_shapes:
            feat_shapes = []
            # 获取各个中间层shape（不含0维），如果含有None则返回默认的feat_shapes
            for l in output[0]:
                if isinstance(l, np.ndarray):
                    shape = l.shape
                else:
                    shape = l.get_shape().as_list()
                shape = shape[1:4]
                if None in shape:
                    feat_shapes = self.params.feat_shapes
                    break
                else:
                    feat_shapes.append(shape)
            self.params = self.params._replace(feat_shapes=feat_shapes)
            sys.stdout.write('[*] Report: variable feat_shapes is {}\n'.format(self.params.feat_shapes))
        return output

    @property
    def anchors(self):
        return self._ssd_anchors_all_layers(self.params.img_shape,
                                            self.params.feat_shapes,
                                            self.params.anchor_sizes,
                                            self.params.anchor_ratios,
                                            self.params.anchor_steps,  # [8, 16, 32, 64, 100, 300]
                                            self.params.anchor_offset)

    def _ssd_net(self, inputs,
                 scope='ssd_net',
                 reuse=False,
                 is_training=True,
                 dropout_keep_prob=0.5):
        with tf.variable_scope(scope, 'ssd_net', [inputs], reuse=reuse) as sc:
            end_points_collection = sc.original_name_scope + '_end_points'
            # Collect outputs for conv2d, fully_connected and max_pool2d.
            with slim.arg_scope(
                    [slim.conv2d, slim.max_pool2d],
                    outputs_collections=end_points_collection):
                end_points = {}
                # ——————————————————Original VGG-16 blocks.———————————————————
                # Block 1.
                net = slim.repeat(inputs, 2, slim.conv2d, 64, [3, 3], scope='conv1')
                end_points['block1'] = net
                net = slim.max_pool2d(net, [2, 2], scope='pool1')
                # Block 2.
                net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], scope='conv2')
                end_points['block2'] = net
                net = slim.max_pool2d(net, [2, 2], scope='pool2')
                # Block 3.
                net = slim.repeat(net, 3, slim.conv2d, 256, [3, 3], scope='conv3')
                end_points['block3'] = net
                net = slim.max_pool2d(net, [2, 2], scope='pool3')
                # Block 4.
                net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv4')
                end_points['block4'] = net
                net = slim.max_pool2d(net, [2, 2], scope='pool4')
                # Block 5.
                net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv5')
                end_points['block5'] = net
                net = slim.max_pool2d(net, [3, 3], stride=1, scope='pool5')
                # ————————————Additional SSD blocks.——————————————————————
                # Block 6
                net = slim.conv2d(net, 1024, [3, 3], rate=6, scope='conv6')
                end_points['block6'] = net
                net = tf.layers.dropout(net, rate=dropout_keep_prob, training=is_training)
                # Block 7
                net = slim.conv2d(net, 1024, [1, 1], scope='conv7')
                end_points['block7'] = net
                net = tf.layers.dropout(net, rate=dropout_keep_prob, training=is_training)
                # Block 8/9/10/11: 1x1 and 3x3 convolutions stride 2 (except lasts).
                end_point = 'block8'
                with tf.variable_scope(end_point):
                    net = slim.conv2d(net, 256, [1, 1], scope='conv1x1')
                    net = tf.pad(net, ([0, 0], [1, 1], [1, 1], [0, 0]), mode='CONSTANT')
                    net = slim.conv2d(net, 512, [3, 3], stride=2, scope='conv3x3', padding='VALID')
                end_points[end_point] = net
                end_point = 'block9'
                with tf.variable_scope(end_point):
                    net = slim.conv2d(net, 128, [1, 1], scope='conv1x1')
                    net = tf.pad(net, ([0, 0], [1, 1], [1, 1], [0, 0]), mode='CONSTANT')
                    net = slim.conv2d(net, 256, [3, 3], stride=2, scope='conv3x3', padding='VALID')
                end_points[end_point] = net
                end_point = 'block10'
                with tf.variable_scope(end_point):
                    net = slim.conv2d(net, 128, [1, 1], scope='conv1x1')
                    net = slim.conv2d(net, 256, [3, 3], scope='conv3x3', padding='VALID')
                end_points[end_point] = net
                end_point = 'block11'
                with tf.variable_scope(end_point):
                    net = slim.conv2d(net, 128, [1, 1], scope='conv1x1')
                    net = slim.conv2d(net, 256, [3, 3], scope='conv3x3', padding='VALID')
                end_points[end_point] = net

                predictions = []
                logits = []
                localisations = []
                # 对于每一feat层进行特征输出
                for i, layer in enumerate(self.default_params.feat_layers):
                    with tf.variable_scope(layer + '_box'):
                        p, l = self._ssd_multibox_layer(end_points[layer],  # <-----SSD处理
                                                        self.params.num_classes,
                                                        self.params.anchor_sizes[i],
                                                        self.params.anchor_ratios[i],
                                                        self.params.normalizations[i])
                    predictions.append(slim.softmax(p))  # prediction_fn=slim.softmax
                    logits.append(p)
                    localisations.append(l)

                import pprint as pp
                from tensorflow.contrib.layers.python.layers import utils
                pp.pprint(end_points)
                end_points_total = utils.convert_collection_to_dict(end_points_collection)
                return predictions, localisations, logits, end_points

    @staticmethod
    def _ssd_arg_scope(weight_decay=0.0005):
        with slim.arg_scope([slim.conv2d, slim.fully_connected],
                            activation_fn=tf.nn.relu,
                            weights_regularizer=slim.l2_regularizer(weight_decay),
                            weights_initializer=tf.contrib.layers.xavier_initializer(),
                            biases_initializer=tf.zeros_initializer()):
            with slim.arg_scope([slim.conv2d, slim.max_pool2d],
                                padding='SAME') as sc:
                return sc

    @staticmethod
    def _ssd_multibox_layer(net,
                            num_classes,
                            sizes,
                            ratios=(1,),
                            normalization=-1):
        # l2 normalize layer
        if normalization > 0:
            scale = tf.Variable(dtype=tf.float32, initial_value=tf.ones(shape=(net.get_shape()[-1],)), trainable=True)
            net = tf.multiply(tf.nn.l2_normalize(net, net.get_shape().ndims-1, epsilon=1e-12), scale)

        # Number of anchors.
        num_anchors = len(sizes) + len(ratios)

        # Location.
        num_loc_pred = num_anchors * 4  # 每一个框有四个坐标
        loc_pred = slim.conv2d(net, num_loc_pred, [3, 3], activation_fn=None,
                               scope='conv_loc')  # 输出C表示不同框的某个坐标
        loc_shape = tensor_shape(loc_pred, rank=4)
        loc_pred = tf.reshape(loc_pred, loc_shape[0:-1]+[loc_shape[-1]//4, 4])

        # Class prediction.
        num_cls_pred = num_anchors * num_classes  # 每一个框都要计算所有的类别
        cls_pred = slim.conv2d(net, num_cls_pred, [3, 3], activation_fn=None,
                               scope='conv_cls')  # 输出C表示不同框的对某个类的预测
        cls_shape = tensor_shape(cls_pred, rank=4)
        cls_pred = tf.reshape(cls_pred, cls_shape[0:-1] + [cls_shape[-1] // num_classes, num_classes])
        return loc_pred, cls_pred

    @staticmethod
    def _ssd_anchors_all_layers(img_shape,
                                layers_shape,
                                anchor_sizes,
                                anchor_ratios,
                                anchor_steps,  # [8, 16, 32, 64, 100, 300]
                                offset=0.5,
                                dtype=np.float32):
        layers_anchors = []
        # 循环生成ssd特征层的搜索网格
        for i, feat_shape in enumerate(layers_shape):
            # 生成feat_shape中HW对应的网格坐标
            y, x = np.mgrid[0:feat_shape[0], 0:feat_shape[1]]
            # step*feat_shape 约等于img_shape，这使得网格点坐标介于0~1，放缩一下即可到图像大小
            y = (y.astype(dtype) + offset) * anchor_steps[i] / img_shape[0]
            x = (x.astype(dtype) + offset) * anchor_steps[i] / img_shape[1]

            # Expand dims to support easy broadcasting.
            y = np.expand_dims(y, axis=-1)
            x = np.expand_dims(x, axis=-1)

            # Compute relative height and width.
            # Tries to follow the original implementation of SSD for the order.
            num_anchors = len(anchor_sizes[i]) + len(anchor_ratios[i])
            h = np.zeros((num_anchors,), dtype=dtype)
            w = np.zeros((num_anchors,), dtype=dtype)
            # Add first anchor boxes with ratio=1.
            h[0] = anchor_sizes[i][0] / img_shape[0]
            w[0] = anchor_sizes[i][0] / img_shape[1]
            di = 1
            if len(anchor_sizes[i]) > 1:
                h[1] = math.sqrt(anchor_sizes[i][0] * anchor_sizes[i][1]) / img_shape[0]
                w[1] = math.sqrt(anchor_sizes[i][0] * anchor_sizes[i][1]) / img_shape[1]
                di += 1
            for i, r in enumerate(anchor_ratios[i]):
                h[i + di] = anchor_sizes[i][0] / img_shape[0] / math.sqrt(r)
                w[i + di] = anchor_sizes[i][0] / img_shape[1] * math.sqrt(r)
            layers_anchors.append((y, x, h, w))

            # 绘制各层中心点示意
            # import matplotlib.pyplot as plt
            # plt.scatter(y, x, c='r', marker='.')
            # plt.grid(True)
            # plt.show()
            # print(h, w)

        return layers_anchors


if __name__ == '__main__':
    ssd = SSDNet()
    ssd.net(tf.placeholder(dtype=tf.float32, shape=[5, 304, 304, 3]))

