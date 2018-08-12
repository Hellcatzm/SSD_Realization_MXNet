import mxnet as mx
import mxnet.ndarray as nd

data_shape = 304
batch_size = 32
rgb_mean = nd.array([123, 117, 104])


def get_iterators(data_shape, batch_size):
    """256, 32"""
    train_iter = mx.image.ImageDetIter(
        batch_size=batch_size,
        data_shape=(3, data_shape, data_shape),
        path_imgrec='./REC_Data/voc2012.rec',
        path_imgidx='./REC_Data/voc2012.idx',
        shuffle=True,
        mean=True,
        rand_crop=1,
        min_object_covered=0.95,
        max_attempts=200)
    # val_iter = image.ImageDetIter(
    #     batch_size=batch_size,
    #     data_shape=(3, data_shape, data_shape),
    #     path_imgrec=data_dir+'val.rec',
    #     shuffle=False,
    #     mean=True)
    return train_iter


# train_data, test_data, class_names, num_class = \
train_data = get_iterators(data_shape, batch_size)
batch = train_data.next()
# (32, 1, 5)
# 1：图像中只有一个目标
# 5：第一个元素对应物体的标号，-1表示非法物体；后面4个元素表示边框，0~1
# 多个目标时list[nd(batch_size, 目标数目, 目标信息)]
print(batch)
# list[nd(batch_size,channel,width,higth)]
print(batch.data[0].shape)
print(batch.data[0])

import matplotlib.pyplot as plt
import matplotlib.patches as patches

# .asnumpy比使用np.array速度快得多
plt.imshow(batch.data[0][2].asnumpy().transpose(1, 2, 0)/255.)

currentAxis = plt.gca()
for i in range(6):
    box = batch.label[0][2][i][1:].asnumpy()*300
    if any(box < 0):
        continue
    print(int(batch.label[0][2][i][0].asscalar()))
    rect = patches.Rectangle((box[1], box[0]), box[3]-box[1], box[2]-box[0],
                             linewidth=1, edgecolor='g', facecolor='none')
    currentAxis.add_patch(rect)
plt.show()

