import mxnet as mx
import numpy as np
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
anchor_sizes = [(21./image_shape[0], 45./image_shape[0]),
                (45./image_shape[0], 99./image_shape[0]),
                (99./image_shape[0], 153./image_shape[0]),
                (153./image_shape[0], 207./image_shape[0]),
                (207./image_shape[0], 261./image_shape[0]),
                (261./image_shape[0], 290./image_shape[0])]
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

    # block = nn.Sequential()
    # block.add(nn.Conv2D(channels=128, kernel_size=1, padding=1, activation='relu'),
    #           nn.Conv2D(channels=256, kernel_size=3, padding=0, activation='relu'))
    # body.add(block)

    class_predictors = nn.Sequential()
    box_predictors = nn.Sequential()
    l2_normalizes = nn.Sequential()
    for i in range(len(feat_layers)):
        if normalizations[i]:
            l2_normalizes.add(L2_normalize(512))
        num_anchors = len(anchor_sizes[i]) + len(anchor_ratios[i]) - 1
        box_predictors.add(nn.Conv2D(num_anchors * 4, 3, padding=1))
        class_predictors.add(nn.Conv2D(num_anchors * (num_classes + 1), 3, padding=1))
    model = nn.Sequential()
    model.add(body, class_predictors, box_predictors, l2_normalizes)
    return model


def ssd_forward(feat, ssd_mod, flatten=True):
    body, class_predictors, box_predictors, l2_normalizes = ssd_mod
    anchors, class_preds, box_preds = ([] for _ in range(3))
    fls = feat_layers.copy()
    fls.insert(0, 0)
    # i为feat_layer索引，j为layer具体层数，feat_layers[i]为特征层
    for i in range(1, len(fls)):
        for j in range(fls[i-1], fls[i]):
            feat = body[j](feat)

        if normalizations[i - 1]:
            feat = l2_normalizes[i-1](feat)
        class_shape = class_predictors[i-1](feat).shape
        box_shape = box_predictors[i-1](feat).shape
        if not flatten:
            class_preds.append(
                class_predictors[i-1](feat).transpose([0, 2, 3, 1]).reshape(
                    class_shape[0], class_shape[2], class_shape[3], class_shape[1]//(num_classes+1), (num_classes+1)))
            box_preds.append(box_predictors[i-1](feat).transpose([0, 2, 3, 1]).reshape(
                box_shape[0], box_shape[2], box_shape[3], box_shape[1]//4, 4))
            # layer_anchors = one_layer_anchers(feat, i - 1)
        else:
            class_preds.append(
                flatten_prediction(class_predictors[i - 1](feat)))
            box_preds.append(
                flatten_prediction(box_predictors[i - 1](feat)))
        # feat_layers索引从1开始，但是其他特征相关索引从0开始
        layer_anchors = nd.contrib.MultiBoxPrior(feat, anchor_sizes[i-1], anchor_ratios[i-1])

        anchors.append(layer_anchors)
    if not flatten:
        return anchors, class_preds, box_preds
    else:
        return concat_predictions(anchors),\
               concat_predictions(class_preds),\
               concat_predictions(box_preds)


class SSDNet(nn.Block):
    def __init__(self, **kwargs):
        super(SSDNet, self).__init__(**kwargs)
        self.model = ssd_model()

    def forward(self, imgs, flatten, *args):
        """
        :param imgs:
        :param flatten:
        :param args:
        :return:
        anchors, 检测框坐标，[1，n，4]
        class_preds, 各图片各检测框分类情况，[bs，n，num_cls+1]
        box_preds, 各图片检测框坐标预测情况，[bs, n*4]
        """
        anchors, class_preds, box_preds = ssd_forward(imgs, self.model, flatten)
        class_preds = class_preds.reshape(shape=(0, -1, num_classes + 1))
        return anchors, class_preds, box_preds


# def one_layer_anchers(feat, i):
#         feat_shape = feat.shape[2:]
#         img_shape = np.array(image_shape)
#         scale = (img_shape//feat_shape)/img_shape
#         y, x = np.mgrid[0:feat_shape[0], 0:feat_shape[1]].astype(np.float32)
#         y = (y + anchor_offset) * scale[0]
#         x = (x + anchor_offset) * scale[1]
#
#         y = np.expand_dims(y, axis=-1)
#         x = np.expand_dims(x, axis=-1)
#
#         num_anchors = len(anchor_sizes[i]) + len(anchor_ratios[i])
#         h = np.zeros((num_anchors,), dtype=float)
#         w = np.zeros((num_anchors,), dtype=float)
#
#         # Add first anchor boxes with ratio=1.
#         h[0] = anchor_sizes[i][0] / img_shape[0]
#         w[0] = anchor_sizes[i][0] / img_shape[1]
#         di = 1
#         if len(anchor_sizes[i]) > 1:
#             h[1] = np.sqrt(anchor_sizes[i][0] * anchor_sizes[i][1]) / img_shape[0]
#             w[1] = np.sqrt(anchor_sizes[i][0] * anchor_sizes[i][1]) / img_shape[1]
#             di += 1
#         for i, r in enumerate(anchor_ratios[i]):
#             h[i + di] = anchor_sizes[i][0] / img_shape[0] / np.sqrt(r)
#             w[i + di] = anchor_sizes[i][0] / img_shape[1] * np.sqrt(r)
#
#         # import matplotlib.pyplot as plt
#         # plt.scatter(y, x, c='r', marker='.')
#         # plt.grid(True)
#         # plt.show()
#         # print(h, w)
#
#         return tuple(nd.array(ar) for ar in (y, x, h, w))


# def bboxes_encode(labels,
#                   bboxes,
#                   anchors,
#                   num_classes=21,
#                   no_annotation_label=21,
#                   ignore_threshold=0.5,
#                   prior_scaling=(0.1, 0.1, 0.2, 0.2)):
#     target_labels = []
#     target_localizations = []
#     target_scores = []
#     # anchors_layer表示单feat层搜索框: (y, x, h, w)
#     for i, anchors_layer in enumerate(anchors):
#         # (b, m, m, k) (b, m, m, k, 4) (b, m, m, k)
#         t_labels, t_loc, t_scores = _bboxes_encode_layer(labels, bboxes,
#                                                          anchors_layer, num_classes,
#                                                          no_annotation_label, ignore_threshold,
#                                                          prior_scaling)
#         target_labels.append(t_labels)
#         target_localizations.append(t_loc)
#         target_scores.append(t_scores)
#     return target_labels, target_localizations, target_scores
#
#
# def _bboxes_encode_layer(labels,  # (b, n,)
#                          bboxes,  # (b, n, 4)
#                          anchors_layer,  # y(m, m, 1), x(m, m, 1), h(k,), w(k,)
#                          num_classes,
#                          no_annotation_label,
#                          ignore_threshold=0.5,
#                          prior_scaling=(0.1, 0.1, 0.2, 0.2)):
#     # Anchors coordinates and volume.
#     yref, xref, href, wref = (nd.array(ar)
#                               for ar in anchors_layer)  # y(m, m, 1), x(m, m, 1), h(k,), w(k,)
#     yref, xref = yref.expand_dims(0), xref.expand_dims(0)  # (1, m, m, k)方便广播
#     ymin = yref - href / 2.  # (m, m, k)
#     xmin = xref - wref / 2.
#     ymax = yref + href / 2.
#     xmax = xref + wref / 2.
#     vol_anchors = (xmax - xmin) * (ymax - ymin)  # 搜索框面积(m, m, k)
#
#     init_type = np.float32
#     shape = (labels.shape[0], yref.shape[1], yref.shape[2], href.size)  # (b, m, m, k)
#     # 初始化标签、得分信息
#     feat_labels = nd.zeros(shape, dtype=init_type)
#     feat_scores = nd.zeros(shape, dtype=init_type)
#     # 初始化坐标信息
#     feat_ymin = nd.zeros(shape, dtype=init_type)
#     feat_xmin = nd.zeros(shape, dtype=init_type)
#     feat_ymax = nd.ones(shape, dtype=init_type)
#     feat_xmax = nd.ones(shape, dtype=init_type)
#
#     def jaccard_with_anchors(bbox):
#         """Compute jaccard score between a box and the anchors.
#         """
#         bs = bbox.shape[0]
#         int_ymin = nd.maximum(ymin, bbox[:, 0].reshape([bs, 1, 1, 1]))  # (b, m, m, k)
#         int_xmin = nd.maximum(xmin, bbox[:, 1].reshape([bs, 1, 1, 1]))
#         int_ymax = nd.minimum(ymax, bbox[:, 2].reshape([bs, 1, 1, 1]))
#         int_xmax = nd.minimum(xmax, bbox[:, 3].reshape([bs, 1, 1, 1]))
#         h = nd.maximum(int_ymax - int_ymin, 0.)
#         w = nd.maximum(int_xmax - int_xmin, 0.)
#         # Volumes.
#         # 处理搜索框和bbox之间的联系
#         inter_vol = h * w  # 交集面积
#         union_vol = vol_anchors - inter_vol + \
#                     ((bbox[:, 2] - bbox[:, 0]) * (bbox[:, 3] - bbox[:, 1])).reshape([bs, 1, 1, 1])  # 并集面积
#         jaccard = inter_vol / union_vol  # 交集/并集，即IOU
#         return jaccard  # (b, m, m, k)
#
#     for i in range(labels.shape[1]):  # 对图像上的obj进行循环
#         # Jaccard score.
#         label = labels[:, i]  # 当前bs图片上第i个对象的标签
#         bbox = bboxes[:, i]  # 当前bs图片上第i个对象的真实框bbox
#
#         jaccard = jaccard_with_anchors(bbox)  # 当前对象的bbox和当前层的搜索网格IOU，(m, m, k)
#         # Mask: check threshold + scores + no annotations + num_classes.
#         mask = nd.greater(jaccard, feat_scores)  # 掩码矩阵，IOU大于历史得分的为True，(b, m, m, k)
#         try:
#             mask = nd.logical_and(mask, (feat_scores > -0.5))
#             mask = nd.logical_and(mask, (label < num_classes).reshape(label.shape[0], 1, 1, 1))  # 不太懂，label应该必定小于类别数
#         except BaseException as e:
#             mask = logical_and(mask, (feat_scores > -0.5))
#             mask = logical_and(mask, (label.reshape(label.shape[0], 1, 1, 1) < num_classes))
#         imask = mask.astype('int32')  # 整形mask
#         fmask = mask.astype('float32')  # 浮点型mask
#
#         # Update values using mask.
#         # 保证feat_labels存储对应位置得分最大对象标签，feat_scores存储那个得分
#         # (m, m, k) × 当前类别scalar + (1 - (m, m, k)) × (m, m, k)
#         # 更新label记录，此时的imask已经保证了True位置当前对像得分高于之前的对象得分，其他位置值不变
#         feat_labels = imask * label.astype('int32').reshape(label.shape[0], 1, 1, 1) +\
#                       (1 - imask) * feat_labels.astype('int32')
#         # 更新score记录，mask为True使用本类别IOU，否则不变
#         feat_scores = nd.where(mask, jaccard, feat_scores)
#
#         # 下面四个矩阵存储对应label的真实框坐标
#         # (b, m, m, k) × 当前框坐标scalar + (1 - (b, m, m, k)) × (b, m, m, k)
#         feat_ymin = fmask * bbox[:, 0].reshape(label.shape[0], 1, 1, 1) + (1 - fmask) * feat_ymin
#         feat_xmin = fmask * bbox[:, 1].reshape(label.shape[0], 1, 1, 1) + (1 - fmask) * feat_xmin
#         feat_ymax = fmask * bbox[:, 2].reshape(label.shape[0], 1, 1, 1) + (1 - fmask) * feat_ymax
#         feat_xmax = fmask * bbox[:, 3].reshape(label.shape[0], 1, 1, 1) + (1 - fmask) * feat_xmax
#
#     # Transform to center / size.
#     # 这里的y、x、h、w指的是对应位置所属真实框的相关属性
#     feat_cy = (feat_ymax + feat_ymin) / 2.  # (b, m, m, k)
#     feat_cx = (feat_xmax + feat_xmin) / 2.
#     feat_h = feat_ymax - feat_ymin
#     feat_w = feat_xmax - feat_xmin
#
#     # Encode features.
#     # prior_scaling: [0.1, 0.1, 0.2, 0.2]，放缩意义不明
#     # ((b, m, m, k) - (m, m, 1)) / (k,) * 10
#     # 以搜索网格中心点为参考，真实框中心的偏移，单位长度为网格hw
#     feat_cy = (feat_cy - yref) / href / prior_scaling[0]  # (b, m, m, k)
#     feat_cx = (feat_cx - xref) / wref / prior_scaling[1]
#     # log((m, m, k) / (m, m, 1)) * 5
#     # 真实框宽高/搜索网格宽高，取对
#     feat_h = nd.log(feat_h / href) / prior_scaling[2]  # (b, m, m, k)
#     feat_w = nd.log(feat_w / wref) / prior_scaling[3]
#     # Use SSD ordering: x / y / w / h instead of ours.(b, m, m, k, 4)
#     feat_localizations = nd.stack(feat_cx, feat_cy, feat_w, feat_h,
#                                   axis=-1)  # -1会扩维，故有4
#
#     # (b, m, m, k) (b, m, m, k, 4) (b, m, m, k)
#     return feat_labels, feat_localizations, feat_scores


# def data_reshape(logits, localisations,  # 预测类别，位置
#                  gclasses, glocalisations, gscores,  # ground truth类别，位置，得分
#                  match_threshold=.5,
#                  negative_ratio=3):
#     lshape = logits[0].shape  # tensor_shape函数可以取代
#     num_classes = lshape[-1]
#     batch_size = lshape[0]
#
#     # Flatten out all vectors!
#     flogits = []
#     fgclasses = []
#     fgscores = []
#     flocalisations = []
#     fglocalisations = []
#     for i in range(len(logits)):  # 按照图片循环
#         flogits.append(nd.reshape(logits[i], [-1, num_classes]))
#         fgclasses.append(nd.reshape(gclasses[i], [-1]))
#         fgscores.append(nd.reshape(gscores[i], [-1]))
#         flocalisations.append(nd.reshape(localisations[i], [-1, 4]))
#         fglocalisations.append(nd.reshape(glocalisations[i], [-1, 4]))
#     # And concat the crap!
#     logits = nd.concat(*flogits, dim=0)  # 全部的搜索框，对应的21类别的输出
#     gclasses = nd.concat(*fgclasses, dim=0)  # 全部的搜索框，真实的类别数字
#     gscores = nd.concat(*fgscores, dim=0)  # 全部的搜索框，和真实框的IOU
#     localisations = nd.concat(*flocalisations, dim=0)
#     glocalisations = nd.concat(*fglocalisations, dim=0)
#
#     # print(logits.shape, gclasses.shape, gscores.shape, localisations.shape, glocalisations.shape)
#
#     dtype = logits.dtype
#     pmask = gscores > match_threshold  # (全部搜索框数目, 21)，类别搜索框和真实框IOU大于阈值
#     fpmask = nd.cast(pmask, dtype)  # 浮点型前景掩码（前景假定为含有对象的IOU足够的搜索框标号）
#     n_positives = nd.sum(fpmask)  # 前景总数
#
#     # Hard negative mining...
#     no_classes = nd.cast(pmask, np.int32)
#     predictions = nd.softmax(logits)  # 此时每一行的21个数转化为概率
#     nmask = logical_and(logical_not(pmask),
#                            gscores > -0.5)  # IOU达不到阈值的类别搜索框位置记1
#     fnmask = nd.cast(nmask, dtype)
#     nvalues = nd.where(nmask,
#                        predictions[:, 0],  # 框内无物体标记为背景预测概率
#                        1. - fnmask)  # 框内有物体位置标记为1
#     nvalues_flat = nd.reshape(nvalues, [-1])
#
#     # Number of negative entries to select.
#     # 在nmask中剔除n_neg个最不可能背景点(对应的class0概率最低)
#     max_neg_entries = nd.cast(nd.sum(fnmask), np.int32)
#     # 3 × 前景掩码数量 + batch_size
#     n_neg = nd.cast(negative_ratio * n_positives, np.int32) + batch_size
#     n_neg = nd.minimum(n_neg, max_neg_entries)
#
#     val, idxes = top_k(-nvalues_flat, k=n_neg.asscalar())  # 最不可能为背景的n_neg个点
#     max_hard_pred = -val[-1]
#     # Final negative mask.
#     nmask = logical_and(nmask, nvalues < max_hard_pred)  # 不是前景，又最不像背景的n_neg个点
#     fnmask = nd.cast(nmask, dtype)
#
#     # print(logits.shape, gclasses.shape, no_classes.shape,
#     #       fpmask.shape, fnmask.shape)
#
#     # p_cls_loss = FocalLoss()
#     # n_cls_loss = FocalLoss()
#     # print(nd.sum(p_cls_loss(logits, gclasses)*fpmask),
#     #       n_cls_loss(logits, no_classes)*fnmask)
#     # reg_loss = SmoothL1Loss()
#     # print(reg_loss(localisations, glocalisations, fpmask.expand_dims(-1)))
#
#     return logits, gclasses, no_classes, fpmask, fnmask, localisations, glocalisations


def training_targets(anchors, class_preds, labels):
    """

    得到的全部边框坐标
    得到的全部边框各个类别得分
    真实类别及对应边框坐标
    :param anchors: 全部的检测框坐标，[1，n，4]
    :param class_preds:当前batch检测框分类结果，[bs，n，num_cls+1(背景0)]
    :param labels:真实数据，[bs，num_obj，5]
    :return:
    box_target, [bs, n*4]
    box_mask, [bs, n*4]
    cls_target, [bs，n]
    """
    class_preds = class_preds.transpose(axes=(0,2,1))
    return nd.contrib.MultiBoxTarget(anchors, labels, class_preds, overlap_threshold=0.5)


if __name__ == '__main__':
    X = mx.ndarray.random.uniform(shape=(1, 1, 304, 304))
    ssd = SSDNet()
    ssd.initialize()
    a, c, b = ssd(X, ssd)
    print(a.shape, c.shape, b.shape)


