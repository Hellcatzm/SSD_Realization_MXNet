import tensorflow as tf
import tfr_data_process
import preprocess_img_tf
import util_tf
from ssd_vgg300_tf import SSDNet
import os

slim = tf.contrib.slim
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

batch_size = 1

# 获取原始数据
image, glabels, gbboxes = \
    tfr_data_process.tfr_read()
# 预处理数据
image, labels, bboxes = \
    preprocess_img_tf.preprocess_image(image, glabels, gbboxes, out_shape=(300, 300))

# 初始化ssd对象
ssd = SSDNet()
# 获取搜索框
ssd_anchors = ssd.anchors

gclasses, glocalisations, gscores = \
    ssd.bboxes_encode(glabels, gbboxes, ssd_anchors)

batch_shape = [1] + [len(ssd_anchors)] * 3  # (1,f层,f层,f层)
# Training batches and queue.
r = tf.train.batch(  # 图片，中心点类别，真实框坐标，得分
    util_tf.reshape_list([image, gclasses, glocalisations, gscores]),
    batch_size=batch_size,
    num_threads=4,
    capacity=5 * batch_size)

batch_queue = slim.prefetch_queue.prefetch_queue(
    r,  # <-----输入格式实际上并不需要调整
    capacity=2 * 1)

# Dequeue batch.
b_image, b_gclasses, b_glocalisations, b_gscores = \
    util_tf.reshape_list(batch_queue.dequeue(), batch_shape)  # 重整list

# Construct SSD network.
# predictions: (BS, H, W, 4, 21)
# localisations: (BS, H, W, 4, 4)
# logits: (BS, H, W, 4, 21)
predictions, localisations, logits, end_points = \
    ssd.net(b_image, is_training=True, weight_decay=0.00004)

from pprint import pprint
pprint([localisations, logits])

# Add loss function.
ssd.losses(logits, localisations,
           b_gclasses, b_glocalisations, b_gscores,
           match_threshold=.5,
           negative_ratio=3,
           alpha=1,
           label_smoothing=.0)
losses = tf.get_collection(tf.GraphKeys.LOSSES)
regularization_losses = tf.get_collection(
                tf.GraphKeys.REGULARIZATION_LOSSES)

regularization_loss = tf.add_n(regularization_losses)
loss = tf.add_n(losses)
sum_loss = tf.add_n([loss, regularization_loss])

global_step = slim.create_global_step()

num_epochs_per_decay = 2.0
num_samples_per_epoch = 17125

adam_beta1 = 0.9
adam_beta2 = 0.999
opt_epsilon = 1.0
moving_average_decay = None

update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
if moving_average_decay:
    moving_average_variables = slim.get_model_variables()
    variable_averages = tf.train.ExponentialMovingAverage(
        moving_average_decay, global_step)
else:
    moving_average_variables, variable_averages = None, None

decay_steps = int(num_samples_per_epoch / batch_size * num_epochs_per_decay)
learning_rate = tf.train.exponential_decay(0.0001,
                                           global_step,
                                           decay_steps,
                                           0.94,  # learning_rate_decay_factor,
                                           staircase=True,
                                           name='exponential_decay_learning_rate')
if moving_average_decay:
    update_ops.append(variable_averages.apply(moving_average_variables))

variables_to_train = tf.trainable_variables()

Optimizer = tf.train.AdamOptimizer(
            learning_rate,
            beta1=adam_beta1,
            beta2=adam_beta2,
            epsilon=opt_epsilon)

grad = Optimizer.compute_gradients(loss, var_list=variables_to_train)
grad_updates = Optimizer.apply_gradients(grad,
                                         global_step=global_step)
update_ops.append(grad_updates)
update_op = tf.group(*update_ops)

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)
config = tf.ConfigProto(log_device_placement=False,
                        gpu_options=gpu_options)
saver = tf.train.Saver(max_to_keep=5,
                       keep_checkpoint_every_n_hours=1.0,
                       write_version=2,
                       pad_step_number=False)


slim.learning.train(update_op,
                    logdir='./logs/',
                    master='',
                    is_chief=True,
                    init_fn=util_tf.get_init_fn(checkpoint_path='./checkpoints/ssd_300_vgg.ckpt',
                                                train_dir='./logs/',
                                                checkpoint_exclude_scopes=None,
                                                checkpoint_model_scope=None,
                                                model_name='ssd_300_vgg',
                                                ignore_missing_vars=False),
                    # summary_op=summary_op,                          # tf.summary.merge节点
                    number_of_steps=None,      # 训练step
                    log_every_n_steps=10,      # 每次model保存step间隔
                    # save_summaries_secs=600,  # 每次summary时间间隔
                    saver=saver,                                    # tf.train.Saver节点
                    save_interval_secs=600,
                    session_config=config,                          # sess参数
                    sync_optimizer=None)


'''
# ————————测试————————
test_program = False
if test_program:
    # import pprint as pp
    # pp.pprint([image, labels, bboxes])

    with tf.Session() as sess:
        print('start runing...')
        init_op = tf.group(tf.global_variables_initializer(),
                           tf.local_variables_initializer())
        sess.run(init_op)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        img, label, boxes = sess.run([image, labels, bboxes])
        coord.request_stop()
        coord.join(threads)

        import matplotlib.pyplot as plt
        import matplotlib.patches as patches
        plt.imshow(img)

        currentAxis = plt.gca()
        for i in range(len(label)):
            box = boxes[i] * 300
            if any(box < 0):
                continue
            print(label[i])
            rect = patches.Rectangle((box[1], box[0]), box[3] - box[1], box[2] - box[0],
                                     linewidth=1, edgecolor='g',
                                     facecolor='none')
            currentAxis.add_patch(rect)

        plt.show()
        print('Done...')
'''

