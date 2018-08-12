## MXNet、TensorFlow双版本实现SSD
`SSD_mini.ipynb`<br>
[『MXNet』第八弹_物体检测之SSD](https://www.cnblogs.com/hellcat/p/9108647.html)<br>
SSD模型原理简介，文件来自MXNet文档，作者应该是李沐，附有个人脚注方便理解。<br>

## 日志
#### 18.8.12
###### 1、提交整理已完成工作
截止目前，完成模块大体如下：
  a、两个框架的压缩数据脚本<br>
    `rec_generate.ipynb`、`tfr_generate.ipynb`<br>
  b、tensorflow压缩数据读取以及预处理（mxnet本身封装的很好，不需自己实现）<br>
    `tfr_data_process.py`、`preprocess_img_tf.py`<br>
  c、两个框架的网络搭建(mxnet需要设定forward逻辑)<br>
    `ssd_vgg300_tf.py`、`ssd_vgg300_mx.py`<br>
  d、辅助函数<br>
    `util_tf.py`、`util_mx.py`
###### 2、mxnet框架l2_normalize层实现
在tf框架中有函数`tf.nn.l2_normalize`（原理自行查阅），SSD中使用如下：
```python
# l2 normalize layer
if normalization > 0:
    scale = tf.Variable(dtype=tf.float32, initial_value=tf.ones(shape=(net.get_shape()[-1],)), trainable=True)
    net = tf.multiply(tf.nn.l2_normalize(net, net.get_shape().ndims-1, epsilon=1e-12), scale)
```
mxnet中并无对应实现，这里实现了l2_normalize并封装上面整个过程为一个新的Block类（位于脚本`util_mx.py`）。
