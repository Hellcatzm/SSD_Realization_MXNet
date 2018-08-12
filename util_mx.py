import mxnet as mx
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


if __name__ == '__main__':
    X = mx.ndarray.array([[1, 2, 3],
                          [1, 2, 3],
                          [1, 2, 3]])
    l2 = L2_normalize(3, axis=1)
    l2.initialize()
    print(l2(X).shape)


