import mxnet as mx
import mxnet.ndarray as nd
import mxnet.image as image
import ssd_vgg300_mx as ssd_mx
import util_mx
import time
import math


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


def test_program():
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches

        ssd = ssd_mx.ssd_model()
        for model in ssd:
            model.initialize()

        data = get_iterators(data_shape=304, batch_size=4)
        batch_data = data.next()
        # 将-1占位符改为背景标签0，对应坐标框记录为[0,0,0,0]
        label = nd.where(batch_data.label[0] < 0,
                         nd.zeros_like(batch_data.label[0]), batch_data.label[0])

        anchors, class_preds, box_preds = ssd_mx.ssd_forward(batch_data.data[0], ssd)
        target_labels, target_localizations, target_scores = \
            ssd_mx.bboxes_encode(label[:, :, 0],
                                 label[:, :, 1:],
                                 anchors)
        # print(label[:, :, 0].shape, label[:, :, 1:].shape)
        # print('target_labels', [f.shape for f in target_labels])
        # print('target_localizations', [f.shape for f in target_localizations])
        # print('target_scores', [f.shape for f in target_scores])
        print('[*] class_preds', [f.shape for f in class_preds])
        print('[*] box_preds', [f.shape for f in box_preds])

        # Add loss function.
        logits, gclasses, no_classes, fpmask, fnmask = \
            ssd_mx.data_reshape(class_preds, box_preds,
                                target_labels, target_localizations, target_scores,
                                match_threshold=.5,
                                negative_ratio=3)

        p_cls_loss = util_mx.FocalLoss()
        n_cls_loss = util_mx.FocalLoss()
        print(nd.sum(p_cls_loss(logits, gclasses) * fpmask),
              nd.sum(n_cls_loss(logits, no_classes) * fnmask))
        reg_loss = util_mx.SmoothL1Loss()
        print(nd.sum(reg_loss(box_preds, target_localizations, fpmask.expand_dims(-1))))

        # .asnumpy比使用np.array速度快得多
        plt.imshow(batch_data.data[0][2].asnumpy().transpose(1, 2, 0) / 255.)
        currentAxis = plt.gca()
        for i in range(6):
            box = batch_data.label[0][2][i][1:].asnumpy() * 300
            if any(box < 0):
                continue
            print(int(batch_data.label[0][2][i][0].asscalar()))
            rect = patches.Rectangle((box[1], box[0]), box[3] - box[1], box[2] - box[0],
                                     linewidth=1, edgecolor='g', facecolor='none')
            currentAxis.add_patch(rect)
        plt.show()


if __name__ == '__main__':
    batch_size = 4
    ctx = mx.cpu(0)
    # ctx = mx.gpu(0)
    # box_metric = mx.MAE()
    cls_metric = mx.metric.Accuracy()
    ssd = ssd_mx.SSDNet()
    ssd.initialize(ctx=ctx)  # mx.init.Xavier(magnitude=2)
    p_cls_loss = util_mx.FocalLoss()
    n_cls_loss = util_mx.FocalLoss()
   # p_cls_loss = gluon.loss.SoftmaxCrossEntropyLoss()
    # n_cls_loss = gluon.loss.SoftmaxCrossEntropyLoss()
    reg_loss = util_mx.SmoothL1Loss()

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
                anchors, class_preds, box_preds = ssd(x, False)


                target_labels, target_localizations, target_scores = \
                    ssd_mx.bboxes_encode(y[:, :, 0],
                                         y[:, :, 1:],
                                         anchors)
                logits, gclasses, no_classes, fpmask, fnmask, localisations, glocalisations = \
                    ssd_mx.data_reshape(class_preds, box_preds,
                                        target_labels, target_localizations, target_scores,
                                        match_threshold=.5,
                                        negative_ratio=3)
                p_cls = nd.sum(p_cls_loss(logits, gclasses) * fpmask)
                n_cls = nd.sum(n_cls_loss(logits, no_classes) * fnmask)
                reg = nd.sum(reg_loss(localisations, glocalisations, fpmask.expand_dims(-1)))
                loss = p_cls + n_cls + reg
                if math.isnan(loss.asscalar()):
                    print(logits, gclasses, fpmask)
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

