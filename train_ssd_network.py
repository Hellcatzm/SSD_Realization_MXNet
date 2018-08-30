import mxnet as mx
import mxnet.ndarray as nd
import mxnet.image as image
import ssd_vgg300_mx as ssd_mx
import util_mx
import time


def get_iterators(data_shape, batch_size):
    train_iter = image.ImageDetIter(
        batch_size=batch_size,
        data_shape=(3, data_shape, data_shape),
        path_imgrec='./REC_Data/voc2012.rec',
        path_imgidx='./REC_Data/voc2012.idx',
        shuffle=True,
        mean=True,
        rand_crop=0,
        min_object_covered=0.95,
        max_attempts=200)
    # val_iter = image.ImageDetIter(
    #     batch_size=batch_size,
    #     data_shape=(3, data_shape, data_shape),
    #     path_imgrec=data_dir+'val.rec',
    #     shuffle=False,
    #     mean=True)
    return train_iter


if __name__ == '__main__':
    batch_size = 4
    ctx = mx.cpu(0)
    # ctx = mx.gpu(0)
    # box_metric = mx.MAE()
    cls_metric = mx.metric.Accuracy()
    ssd = ssd_mx.SSDNet()
    ssd.initialize(ctx=ctx)  # mx.init.Xavier(magnitude=2)

    cls_loss = util_mx.FocalLoss()
    box_loss = util_mx.SmoothL1Loss()

    trainer = mx.gluon.Trainer(ssd.collect_params(),
                               'sgd', {'learning_rate': 0.01, 'wd': 5e-4})

    data = get_iterators(data_shape=304, batch_size=batch_size)
    for epoch in range(30):
        # reset data iterators and metrics
        data.reset()
        cls_metric.reset()
        # box_metric.reset()
        tic = time.time()
        for i, batch in enumerate(data):
            start_time = time.time()
            x = batch.data[0].as_in_context(ctx)
            y = batch.label[0].as_in_context(ctx)
            # 将-1占位符改为背景标签0，对应坐标框记录为[0,0,0,0]
            y = nd.where(y < 0, nd.zeros_like(y), y)
            with mx.autograd.record():
                # anchors, 检测框坐标，[1，n，4]
                # class_preds, 各图片各检测框分类情况，[bs，n，num_cls + 1]
                # box_preds, 各图片检测框坐标预测情况，[bs, n * 4]
                anchors, class_preds, box_preds = ssd(x, True)

                # box_target, 检测框的收敛目标，[bs, n * 4]
                # box_mask, 隐藏不需要的背景类，[bs, n * 4]
                # cls_target, 记录全检测框的真实类别，[bs，n]
                box_target, box_mask, cls_target = ssd_mx.training_targets(anchors, class_preds, y)

                loss1 = cls_loss(class_preds, cls_target)
                loss2 = box_loss(box_preds, box_target, box_mask)
                loss = loss1 + loss2
            loss.backward()
            trainer.step(batch_size)
            if i % 1 == 0:
                duration = time.time() - start_time
                examples_per_sec = batch_size / duration
                sec_per_batch = float(duration)
                format_str = "[*] step %d,  loss=%.2f (%.1f examples/sec; %.3f sec/batch)"
                print(format_str % (i, nd.sum(loss).asscalar(), examples_per_sec, sec_per_batch))
            if i % 500 == 0:
                ssd.model.save_parameters('model_mx_{}.params'.format(epoch))