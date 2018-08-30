## MXNet实现SSD
`SSD_mini.ipynb`<br>
[『MXNet』第八弹_物体检测之SSD](https://www.cnblogs.com/hellcat/p/9108647.html)<br>
SSD模型原理简介，文件来自MXNet文档，作者应该是李沐，附有个人脚注方便理解。<br>

## 运行程序
#### 数据准备
将VOC2012数据解压到文件夹`VOC2012`中，注意检查下一级目录包含`Annotations`文件夹和`JPEGImages`文件夹。
#### 生成压缩文件
使用jupyter打开`rec_generate.ipynb`按照顺序运行即可。
REC压缩文件处理相关介绍见：<br>
[『MXNet』im2rec脚本使用以及数据读取](https://www.cnblogs.com/hellcat/p/9373890.html)
#### 训练
```python
python train_ssd_network.py
```

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
#### 18.8.16
###### 工作框架搭建完毕
两个版本的ssd已经可以正常训练，但是效果未加验明，且mxnet版本没有添加存储模块。
#### 18.8.21
###### 删除tensorflow相关内容
现在这是一个单纯的MXNet框架项目，并添加了保存模型的语句。
#### 18.8.30
###### 重新整理框架内容
初版参照tensorflow版本过多，实际训练不能收敛。可能是两个框架流程差异（如mx的各个图片标签长度必须一致，tf图像在送入神经网络之前都是不计算bs维度的等）导致我的修改较为繁琐，增加出错几率，这里更多的使用了原汁原味的mxnet逻辑及封装，现已可以正常训练，旧版代码并未删除（因为注释较多，虽运行有问题，但对会议理解ssd工作原理还是有好处的）。<br>
`train_ssd_network_old.py`用于保存原来的调用代码，`train_ssd_network.py`是新的训练程序入口。<br>
`SSD_mini.ipynb`中有关于eval的代码，加上读取参数部分后可以直接拿来用。
