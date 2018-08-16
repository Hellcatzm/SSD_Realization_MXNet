import mxnet as mx
import numpy as np
from mxnet import gluon
from mxnet.gluon import nn
import mxnet.ndarray as nd


def repeat(num_convs, num_channel, pool=True):
    block = nn.Sequential()
    if pool:
        block.add(nn.MaxPool2D(pool_size=2, strides=2))
    [block.add(
        nn.Conv2D(num_channel, kernel_size=3, padding=1, activation='relu'))
     for _ in range(num_convs)]
    return block


class L2_normalize(nn.Block):
    def __init__(self, feat_channels, axis=1, epslion=1e-12, **kwargs):
        """
        L2正则化可训练封装，参考tf.nn.l2_normalize实现
        :param feat_channels: channels数目，是axis所在维度的数目
        :param axis: axis表示channel所在维度，必须是1，否则forward最后一步的转置需要调整
        :param epslion: 防止分母为0，没必要改动
        :param kwargs:
        """
        super(L2_normalize, self).__init__(**kwargs)
        self.axis = axis
        self.epsilon = epslion
        self.scale = self.params.get('scale', shape=(feat_channels,))

    def forward(self, feat):
        square_sum = nd.sum(nd.square(feat), axis=self.axis, keepdims=True)
        inv_norm = nd.rsqrt(nd.maximum(square_sum, self.epsilon))
        l2_res = nd.multiply(feat, inv_norm)
        print(l2_res.shape)
        return nd.multiply(l2_res.transpose([0, 2, 3, 1]), self.scale.data()).transpose([0, 3, 1, 2])


def logical_and(a, b):
    ctx = a.context
    a = a.asnumpy()
    b = b.asnumpy()
    return nd.array(np.logical_and(a, b), ctx=ctx)


def logical_not(a):
    ctx = a.context
    a = a.asnumpy()
    return nd.array(np.logical_not(a), ctx=ctx)


def top_k(array, k):
    s = nd.argsort(array)[::-1][:k]
    return array[s], s


class FocalLoss(gluon.loss.Loss):
    def __init__(self, axis=-1, alpha=0.25, gamma=2, batch_axis=0, **kwargs):
        super(FocalLoss, self).__init__(None, batch_axis, **kwargs)
        self._axis = axis
        self._alpha = alpha
        self._gamma = gamma

    def hybrid_forward(self, F, output, label):
        output = F.softmax(output)
        pj = output.pick(label, axis=self._axis, keepdims=True)
        loss = - self._alpha * ((1 - pj) ** self._gamma) * pj.log()
        return loss.mean(axis=self._batch_axis, exclude=True)


class SmoothL1Loss(gluon.loss.Loss):
    def __init__(self, batch_axis=0, **kwargs):
        super(SmoothL1Loss, self).__init__(None, batch_axis, **kwargs)

    def hybrid_forward(self, F, output, label, mask):
        # 参数mask，屏蔽掉不需要被惩罚的负例样本
        loss = F.smooth_l1((output - label) * mask, scalar=1.0)
        return loss.mean(self._batch_axis, exclude=True)


if __name__ == '__main__':
    cls_loss = FocalLoss()
    print(cls_loss)

