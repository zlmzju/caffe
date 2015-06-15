#!/usr/bin/env python
import numpy as np
import caffe
#ROOT='/home/liming/project/caffe/examples/finetune_fcn/'
MODEL_FILE = '../../train/MSRA9000/deploy.prototxt'
PRETRAINED = '../../train/MSRA9000/models0/train_iter_100000.caffemodel'
caffe.set_device(1)
caffe.set_mode_gpu()
net = caffe.Classifier(MODEL_FILE, PRETRAINED,caffe.TEST)
print('\nblobs')
print [(k, v.data.shape) for k, v in net.blobs.items()]
#print('\nparams')
#print [(k, v[0].data.shape) for k, v in net.params.items()]

NEWMODEL_FILE = '../deploy.prototxt'
NEWPRETRAINED = PRETRAINED
finetune_net = caffe.Classifier(NEWMODEL_FILE, NEWPRETRAINED,
                       caffe.TEST)
#conv8=np.squeeze(finetune_net.params['conv8'][0].data)
#print('\nparams')
#print [(k, v[0].data.shape) for k, v in finetune_net.params.items()]
#finetune_net.params['conv8'][0].data.flat=net.params['conv8'][0].data.flat
#finetune_net.params['conv8'][1].data.flat=net.params['conv8'][1].data.flat
#finetune_net.params['deconv'][0].data.flat=net.params['deconv'][0].data.flat
#finetune_net.params['deconv'][1].data.flat=net.params['deconv'][1].data.flat
#finetune_net.save('../guidedInitModel.caffemodel')


##test the net
#transformer = caffe.io.Transformer({'data': finetune_net.blobs['data'].data.shape})
#transformer.set_transpose('data', (2,0,1))
#transformer.set_mean('data', np.load(caffe_root + 'python/caffe/imagenet/ilsvrc_2012_mean.npy').mean(1).mean(1)) # mean pixel
#transformer.set_raw_scale('data', 255)  # the reference model operates on images in [0,255] range instead of [0,1]
#transformer.set_channel_swap('data', (2,1,0))  # the reference model has channels in BGR order instead of RGB
#finetune_net.blobs['data'].reshape(1,3,500,500)
#finetune_net.blobs['data'].data[...] = transformer.preprocess('data', caffe.io.load_image(caffe_root+'examples/images/cat.jpg'))
#
#out=finetune_net.forward()
#outmap=np.squeeze(out['map'])
#plt.imshow(outmap)