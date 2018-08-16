import mxnet as mx
from mxnet.gluon import nn


def repeat(num_convs, num_channel, pool=True):
    block = nn.Sequential()
    if pool:
        block.add(nn.MaxPool2D(pool_size=2, strides=2))
    [block.add(
        nn.Conv2D(num_channel, kernel_size=3, padding=1, activation='relu'))
     for _ in range(num_convs)]
    return block


class SSD(nn.Block):
    def __init__(self, conv_arch, dropout_keep_prob, **kwargs):
        super(SSD, self).__init__(**kwargs)
        self.vgg_conv = nn.Sequential()
        self.vgg_conv.add(repeat(*conv_arch[0], pool=False))
        [self.vgg_conv.add(repeat(*conv_arch[i])) for i in range(1, len(conv_arch))]
        # 迭代器对象只能进行单次迭代，所以将之转化为tuple，否则识别参数处迭代后forward再次迭代直接跳出循环
        # self.vgg_conv = tuple([repeat(*conv_arch[i])
        #                       for i in range(len(conv_arch))])
        # 只能识别实例属性直接为mx层函数或者mx序列对象的参数，如果使用其他容器，需要将参数收集进参数字典
        # _ = [self.params.update(block.collect_params()) for block in self.vgg_conv]

        self.block6 = nn.Conv2D(channels=1024, kernel_size=3, padding=1, activation='relu')
        self.block7 = nn.Sequential()
        self.block7.add(nn.Dropout(rate=dropout_keep_prob),
                        nn.Conv2D(channels=1024, kernel_size=1, padding=0, activation='relu'))
        self.block8 = nn.Sequential()
        self.block8.add(nn.Dropout(rate=dropout_keep_prob),
                        nn.Conv2D(channels=256, kernel_size=1, padding=0, activation='relu'),
                        nn.Conv2D(channels=512, kernel_size=3, strides=2, padding=1, activation='relu'))
        self.block9 = nn.Sequential()
        self.block9.add(nn.Conv2D(channels=256, kernel_size=1, padding=0, activation='relu'),
                        nn.Conv2D(channels=512, kernel_size=3, strides=2, padding=1, activation='relu'))
        self.block10 = nn.Sequential()
        self.block10.add(nn.Conv2D(channels=128, kernel_size=1, padding=0, activation='relu'),
                         nn.Conv2D(channels=256, kernel_size=3,  strides=2, padding=1, activation='relu'))
        self.block11 = nn.Sequential()
        self.block11.add(nn.Conv2D(channels=128, kernel_size=1, padding=0, activation='relu'),
                         nn.Conv2D(channels=256, kernel_size=3, padding=0, activation='relu'))
        self.block12 = nn.Sequential()
        self.block12.add(nn.Conv2D(channels=128, kernel_size=1, padding=1, activation='relu'),
                         nn.Conv2D(channels=256, kernel_size=3, padding=0, activation='relu'))

    def forward(self, x):
        end_points = {'block0': x}
        for (index, block) in enumerate(self.vgg_conv):
            end_points.update({'block{:d}'.format(index+1): block(end_points['block{:d}'.format(index)])})

        end_points['block6'] = self.block6(end_points['block5'])
        end_points['block7'] = self.block7(end_points['block6'])
        end_points['block8'] = self.block8(end_points['block7'])
        end_points['block9'] = self.block9(end_points['block8'])
        end_points['block10'] = self.block10(end_points['block9'])
        end_points['block11'] = self.block11(end_points['block10'])
        end_points['block12'] = self.block12(end_points['block11'])

        return end_points


if __name__ == '__main__':

    ssd = SSD(conv_arch=((2, 64), (2, 128), (3, 256), (3, 512), (3, 512)),
              dropout_keep_prob=0.5)
    ssd.initialize()
    X = mx.ndarray.random.uniform(shape=(1, 1, 304, 304))
    import pprint as pp
    pp.pprint([x[1].shape for x in ssd(X).items()])
    '''
    {'block1': <tf.Tensor 'ssd_net/conv1/conv1_2/Relu:0' shape=(5, 304, 304, 64) dtype=float32>,
     'block2': <tf.Tensor 'ssd_net/conv2/conv2_2/Relu:0' shape=(5, 152, 152, 128) dtype=float32>,
     'block3': <tf.Tensor 'ssd_net/conv3/conv3_3/Relu:0' shape=(5, 76, 76, 256) dtype=float32>,
     'block4': <tf.Tensor 'ssd_net/conv4/conv4_3/Relu:0' shape=(5, 38, 38, 512) dtype=float32>, //
     'block5': <tf.Tensor 'ssd_net/conv5/conv5_3/Relu:0' shape=(5, 19, 19, 512) dtype=float32>,
     'block6': <tf.Tensor 'ssd_net/conv6/Relu:0' shape=(5, 19, 19, 1024) dtype=float32>,
     'block7': <tf.Tensor 'ssd_net/conv7/Relu:0' shape=(5, 19, 19, 1024) dtype=float32>, //
     'block8': <tf.Tensor 'ssd_net/block8/conv3x3/Relu:0' shape=(5, 10, 10, 512) dtype=float32>, //
     'block9': <tf.Tensor 'ssd_net/block9/conv3x3/Relu:0' shape=(5, 5, 5, 256) dtype=float32>, //
     'block10': <tf.Tensor 'ssd_net/block10/conv3x3/Relu:0' shape=(5, 3, 3, 256) dtype=float32>, //
     'block11': <tf.Tensor 'ssd_net/block11/conv3x3/Relu:0' shape=(5, 1, 1, 256) dtype=float32>} //
    '''
