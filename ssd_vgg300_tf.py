import sys
import math
import numpy as np
import tensorflow as tf
from collections import namedtuple

from util_tf import *

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

    def net(self, input_data, weight_decay, update_feat_shapes=True, is_training=True):
        with slim.arg_scope(self._ssd_arg_scope(weight_decay)):
            output = self._ssd_net(input_data, is_training=is_training)
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

                # import pprint as pp
                # from tensorflow.contrib.layers.python.layers import utils
                # pp.pprint(end_points)
                end_points_total = slim.utils.convert_collection_to_dict(end_points_collection)
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
        return cls_pred, loc_pred

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

    def bboxes_encode(self, labels, bboxes, anchors, scope=None):
        return tf_ssd_bboxes_encode(
            labels, bboxes, anchors,
            self.params.num_classes,
            self.params.no_annotation_label,  # 21
            ignore_threshold=0.5,
            prior_scaling=self.params.prior_scaling,
            scope=scope)

    @staticmethod
    def losses(logits, localisations,
               gclasses, glocalisations, gscores,
               match_threshold=0.5,
               negative_ratio=3.,
               alpha=1.,
               label_smoothing=0.,
               scope='ssd_losses'):
        """Define the SSD network losses.
        """
        return ssd_losses(logits, localisations,
                          gclasses, glocalisations, gscores,
                          match_threshold=match_threshold,
                          negative_ratio=negative_ratio,
                          alpha=alpha,
                          label_smoothing=label_smoothing,
                          scope=scope)


def tf_ssd_bboxes_encode(labels,
                         bboxes,
                         anchors,
                         num_classes,
                         no_annotation_label,
                         ignore_threshold=0.5,
                         prior_scaling=(0.1, 0.1, 0.2, 0.2),
                         dtype=tf.float32,
                         scope='ssd_bboxes_encode'):
    with tf.name_scope(scope):
        target_labels = []
        target_localizations = []
        target_scores = []
        # anchors_layer: (y, x, h, w)
        for i, anchors_layer in enumerate(anchors):
            with tf.name_scope('bboxes_encode_block_%i' % i):
                # (m,m,k)，xywh(m,m,4k)，(m,m,k)
                t_labels, t_loc, t_scores = \
                    tf_ssd_bboxes_encode_layer(labels, bboxes, anchors_layer,
                                               num_classes, no_annotation_label,
                                               ignore_threshold,
                                               prior_scaling, dtype)
                target_labels.append(t_labels)
                target_localizations.append(t_loc)
                target_scores.append(t_scores)
        return target_labels, target_localizations, target_scores


# 为了有助理解，m表示该层中心点行列数，k为每个中心点对应的框数，n为图像上的目标数
def tf_ssd_bboxes_encode_layer(labels,  # (n,)
                               bboxes,  # (n, 4)
                               anchors_layer,  # y(m, m, 1), x(m, m, 1), h(k,), w(k,)
                               num_classes,
                               no_annotation_label,
                               ignore_threshold=0.5,
                               prior_scaling=(0.1, 0.1, 0.2, 0.2),
                               dtype=tf.float32):
    """Encode groundtruth labels and bounding boxes using SSD anchors from
    one layer.

    Arguments:
      labels: 1D Tensor(int64) containing groundtruth labels;
      bboxes: Nx4 Tensor(float) with bboxes relative coordinates;
      anchors_layer: Numpy array with layer anchors;
      matching_threshold: Threshold for positive match with groundtruth bboxes;
      prior_scaling: Scaling of encoded coordinates.

    Return:
      (target_labels, target_localizations, target_scores): Target Tensors.
    """
    # Anchors coordinates and volume.
    yref, xref, href, wref = anchors_layer  # y(m, m, 1), x(m, m, 1), h(k,), w(k,)
    ymin = yref - href / 2.  # (m, m, k)
    xmin = xref - wref / 2.
    ymax = yref + href / 2.
    xmax = xref + wref / 2.
    vol_anchors = (xmax - xmin) * (ymax - ymin)  # 搜索框面积(m, m, k)

    # Initialize tensors...
    # 下面各个Tensor矩阵的shape等于中心点坐标矩阵的shape
    shape = (yref.shape[0], yref.shape[1], href.size)  # (m, m, k)
    feat_labels = tf.zeros(shape, dtype=tf.int64)  # (m, m, k)
    feat_scores = tf.zeros(shape, dtype=dtype)

    feat_ymin = tf.zeros(shape, dtype=dtype)
    feat_xmin = tf.zeros(shape, dtype=dtype)
    feat_ymax = tf.ones(shape, dtype=dtype)
    feat_xmax = tf.ones(shape, dtype=dtype)

    def jaccard_with_anchors(bbox):
        """Compute jaccard score between a box and the anchors.
        """
        int_ymin = tf.maximum(ymin, bbox[0])  # (m, m, k)
        int_xmin = tf.maximum(xmin, bbox[1])
        int_ymax = tf.minimum(ymax, bbox[2])
        int_xmax = tf.minimum(xmax, bbox[3])
        h = tf.maximum(int_ymax - int_ymin, 0.)
        w = tf.maximum(int_xmax - int_xmin, 0.)
        # Volumes.
        # 处理搜索框和bbox之间的联系
        inter_vol = h * w  # 交集面积
        union_vol = vol_anchors - inter_vol \
                    + (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])  # 并集面积
        jaccard = tf.div(inter_vol, union_vol)  # 交集/并集，即IOU
        return jaccard  # (m, m, k)

    def condition(i, feat_labels, feat_scores,
                  feat_ymin, feat_xmin, feat_ymax, feat_xmax):
        """Condition: check label index.
        """
        r = tf.less(i, tf.shape(labels))
        return r[0]  # tf.shape(labels)有维度，所以r有维度

    def body(i, feat_labels, feat_scores,
             feat_ymin, feat_xmin, feat_ymax, feat_xmax):
        """Body: update feature labels, scores and bboxes.
        Follow the original SSD paper for that purpose:
          - assign values when jaccard > 0.5;
          - only update if beat the score of other bboxes.
        """
        # Jaccard score.
        label = labels[i]  # 当前图片上第i个对象的标签
        bbox = bboxes[i]  # 当前图片上第i个对象的真实框bbox
        jaccard = jaccard_with_anchors(bbox)  # 当前对象的bbox和当前层的搜索网格IOU，(m, m, k)
        # Mask: check threshold + scores + no annotations + num_classes.
        mask = tf.greater(jaccard, feat_scores)  # 掩码矩阵，IOU大于历史得分的为True，(m, m, k)
        # mask = tf.logical_and(mask, tf.greater(jaccard, matching_threshold))
        mask = tf.logical_and(mask, feat_scores > -0.5)
        mask = tf.logical_and(mask, label < num_classes)  # 不太懂，label应该必定小于类别数
        imask = tf.cast(mask, tf.int64)  # 整形mask
        fmask = tf.cast(mask, dtype)  # 浮点型mask

        # Update values using mask.
        # 保证feat_labels存储对应位置得分最大对象标签，feat_scores存储那个得分
        # (m, m, k) × 当前类别scalar + (1 - (m, m, k)) × (m, m, k)
        # 更新label记录，此时的imask已经保证了True位置当前对像得分高于之前的对象得分，其他位置值不变
        feat_labels = imask * label + (1 - imask) * feat_labels
        # 更新score记录，mask为True使用本类别IOU，否则不变
        feat_scores = tf.where(mask, jaccard, feat_scores)

        # 下面四个矩阵存储对应label的真实框坐标
        # (m, m, k) × 当前框坐标scalar + (1 - (m, m, k)) × (m, m, k)
        feat_ymin = fmask * bbox[0] + (1 - fmask) * feat_ymin
        feat_xmin = fmask * bbox[1] + (1 - fmask) * feat_xmin
        feat_ymax = fmask * bbox[2] + (1 - fmask) * feat_ymax
        feat_xmax = fmask * bbox[3] + (1 - fmask) * feat_xmax

        return [i + 1, feat_labels, feat_scores,
                feat_ymin, feat_xmin, feat_ymax, feat_xmax]

    # Main loop definition.
    # 对当前图像上每一个目标进行循环
    i = 0
    (i,
     feat_labels, feat_scores,
     feat_ymin, feat_xmin,
     feat_ymax, feat_xmax) = tf.while_loop(condition, body,
                                           [i,
                                            feat_labels, feat_scores,
                                            feat_ymin, feat_xmin,
                                            feat_ymax, feat_xmax])
    # Transform to center / size.
    # 这里的y、x、h、w指的是对应位置所属真实框的相关属性
    feat_cy = (feat_ymax + feat_ymin) / 2.
    feat_cx = (feat_xmax + feat_xmin) / 2.
    feat_h = feat_ymax - feat_ymin
    feat_w = feat_xmax - feat_xmin

    # Encode features.
    # prior_scaling: [0.1, 0.1, 0.2, 0.2]，放缩意义不明
    # ((m, m, k) - (m, m, 1)) / (k,) * 10
    # 以搜索网格中心点为参考，真实框中心的偏移，单位长度为网格hw
    feat_cy = (feat_cy - yref) / href / prior_scaling[0]
    feat_cx = (feat_cx - xref) / wref / prior_scaling[1]
    # log((m, m, k) / (m, m, 1)) * 5
    # 真实框宽高/搜索网格宽高，取对
    feat_h = tf.log(feat_h / href) / prior_scaling[2]
    feat_w = tf.log(feat_w / wref) / prior_scaling[3]
    # Use SSD ordering: x / y / w / h instead of ours.(m, m, k, 4)
    feat_localizations = tf.stack([feat_cx, feat_cy, feat_w, feat_h], axis=-1)  # -1会扩维，故有4

    return feat_labels, feat_localizations, feat_scores


def ssd_losses(logits, localisations,  # 预测类别，位置
               gclasses, glocalisations, gscores,  # ground truth类别，位置，得分
               match_threshold=0.5,
               negative_ratio=3.,
               alpha=1.,
               label_smoothing=0.,
               scope=None):
    with tf.name_scope(scope, 'ssd_losses'):
        # 提取类别数和batch_size
        lshape = tensor_shape(logits[0], 5)  # tensor_shape函数可以取代
        num_classes = lshape[-1]
        batch_size = lshape[0]

        # Flatten out all vectors!
        flogits = []
        fgclasses = []
        fgscores = []
        flocalisations = []
        fglocalisations = []
        for i in range(len(logits)):  # 按照图片循环
            flogits.append(tf.reshape(logits[i], [-1, num_classes]))
            fgclasses.append(tf.reshape(gclasses[i], [-1]))
            fgscores.append(tf.reshape(gscores[i], [-1]))
            flocalisations.append(tf.reshape(localisations[i], [-1, 4]))
            fglocalisations.append(tf.reshape(glocalisations[i], [-1, 4]))
        # And concat the crap!
        logits = tf.concat(flogits, axis=0)  # 全部的搜索框，对应的21类别的输出
        gclasses = tf.concat(fgclasses, axis=0)  # 全部的搜索框，真实的类别数字
        gscores = tf.concat(fgscores, axis=0)  # 全部的搜索框，和真实框的IOU
        localisations = tf.concat(flocalisations, axis=0)
        glocalisations = tf.concat(fglocalisations, axis=0)

        """[<tf.Tensor 'ssd_losses/concat:0' shape=(279424, 21) dtype=float32>,
            <tf.Tensor 'ssd_losses/concat_1:0' shape=(279424,) dtype=int64>,
            <tf.Tensor 'ssd_losses/concat_2:0' shape=(279424,) dtype=float32>,
            <tf.Tensor 'ssd_losses/concat_3:0' shape=(279424, 4) dtype=float32>,
            <tf.Tensor 'ssd_losses/concat_4:0' shape=(279424, 4) dtype=float32>]
        """

        dtype = logits.dtype
        pmask = gscores > match_threshold  # (全部搜索框数目, 21)，类别搜索框和真实框IOU大于阈值
        fpmask = tf.cast(pmask, dtype)  # 浮点型前景掩码（前景假定为含有对象的IOU足够的搜索框标号）
        n_positives = tf.reduce_sum(fpmask)  # 前景总数

        # Hard negative mining...
        no_classes = tf.cast(pmask, tf.int32)
        predictions = slim.softmax(logits)  # 此时每一行的21个数转化为概率
        nmask = tf.logical_and(tf.logical_not(pmask),
                               gscores > -0.5)  # IOU达不到阈值的类别搜索框位置记1
        fnmask = tf.cast(nmask, dtype)
        nvalues = tf.where(nmask,
                           predictions[:, 0],  # 框内无物体标记为背景预测概率
                           1. - fnmask)  # 框内有物体位置标记为1
        nvalues_flat = tf.reshape(nvalues, [-1])

        # Number of negative entries to select.
        # 在nmask中剔除n_neg个最不可能背景点(对应的class0概率最低)
        max_neg_entries = tf.cast(tf.reduce_sum(fnmask), tf.int32)
        # 3 × 前景掩码数量 + batch_size
        n_neg = tf.cast(negative_ratio * n_positives, tf.int32) + batch_size
        n_neg = tf.minimum(n_neg, max_neg_entries)
        val, idxes = tf.nn.top_k(-nvalues_flat, k=n_neg)  # 最不可能为背景的n_neg个点
        max_hard_pred = -val[-1]
        # Final negative mask.
        nmask = tf.logical_and(nmask, nvalues < max_hard_pred)  # 不是前景，又最不像背景的n_neg个点
        fnmask = tf.cast(nmask, dtype)

        # Add cross-entropy loss.
        with tf.name_scope('cross_entropy_pos'):
            loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,
                                                                  labels=gclasses)  # 0-20
            loss = tf.div(tf.reduce_sum(loss * fpmask), batch_size, name='value')
            tf.losses.add_loss(loss)

        with tf.name_scope('cross_entropy_neg'):
            loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,
                                                                  labels=no_classes)  # {0,1}
            loss = tf.div(tf.reduce_sum(loss * fnmask), batch_size, name='value')
            tf.losses.add_loss(loss)

        # Add localization loss: smooth L1, L2, ...
        with tf.name_scope('localization'):
            # Weights Tensor: positive mask + random negative.
            weights = tf.expand_dims(alpha * fpmask, axis=-1)
            loss = abs_smooth(localisations - glocalisations)
            loss = tf.div(tf.reduce_sum(loss * weights), batch_size, name='value')
            tf.losses.add_loss(loss)


if __name__ == '__main__':
    ssd = SSDNet()
    ssd.net(tf.placeholder(dtype=tf.float32, shape=[5, 304, 304, 3]))

