import tensorflow as tf
import tfr_data_process
import preprocess_img_tf
from ssd_vgg300_tf import SSDNet

# 获取原始数据
image, glabels, gbboxes = \
    tfr_data_process.tfr_read()
# 预处理数据
image, labels, bboxes = \
    preprocess_img_tf.preprocess_image(image, glabels, gbboxes, out_shape=(300, 300))

# 初始化ssd对象
ssd = SSDNet()
# 获取搜索框
anchors = ssd.anchors

# ##########测试##########
import pprint as pp
pp.pprint([image, labels, bboxes])

with tf.Session() as sess:
    print('start runing...')
    img = sess.run(image)
    import matplotlib.pyplot as plt
    plt.imshow(img)
    plt.show()
    print('Done...')
