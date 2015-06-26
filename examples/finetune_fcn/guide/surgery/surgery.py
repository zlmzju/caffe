#!/usr/bin/env python
import numpy as np
import caffe
#ROOT='/home/liming/project/caffe/examples/finetune_fcn/'
MODEL_FILE = '../../train/MSRA9000/deploy.prototxt'
PRETRAINED = '../../train/MSRA9000/models0/train_iter_100000.caffemodel'
caffe.set_device(1)
caffe.set_mode_gpu()
net = caffe.Classifier(MODEL_FILE, PRETRAINED,caffe.TEST)
print('\nblobs net')
print [(k, v.data.shape) for k, v in net.blobs.items()]
print('\nparams net')
print [(k, v[0].data.shape) for k, v in net.params.items()]

NEWMODEL_FILE = '../deploy.prototxt'
NEWPRETRAINED = PRETRAINED
finetune_net = caffe.Classifier(NEWMODEL_FILE, NEWPRETRAINED,
                       caffe.TEST)
print('\nblobs finetune_net')
print [(k, v.data.shape) for k, v in finetune_net.blobs.items()]
print('\nparams finetune_net')
print [(k, v[0].data.shape) for k, v in finetune_net.params.items()]
#piecewise constant interpolation
#finetune_net.params['interp'][0].data.flat=np.ones(finetune_net.params['interp'][0].data.shape)
#finetune_net.params['interp'][1].data.flat=np.zeros(finetune_net.params['interp'][1].data.shape)
#grayscal or intensity = 0.2989 * R + 0.5870 * G + 0.1140 * B 
finetune_net.params['conv_edge1'][0].data[:,0,:].flat=0.1140*np.ones(finetune_net.params['conv_edge1'][0].data.shape) #* finetune_net.params['conv_edge1'][0].data[:,0,:]
finetune_net.params['conv_edge1'][0].data[:,1,:].flat=0.5870*np.ones(finetune_net.params['conv_edge1'][0].data.shape) #* finetune_net.params['conv_edge1'][0].data[:,1,:]
finetune_net.params['conv_edge1'][0].data[:,2,:].flat=0.2989*np.ones(finetune_net.params['conv_edge1'][0].data.shape) #* finetune_net.params['conv_edge1'][0].data[:,2,:]
finetune_net.params['conv_edge1'][1].data.flat=0*np.ones(finetune_net.params['conv_edge1'][1].data.shape)

finetune_net.params['conv_smooth1'][0].data.flat=1*np.ones(finetune_net.params['conv_smooth1'][0].data.shape)
finetune_net.params['conv_smooth1'][1].data.flat=0*np.ones(finetune_net.params['conv_smooth1'][1].data.shape)

finetune_net.params['conv_combine'][0].data.flat=1*np.ones(finetune_net.params['conv_combine'][0].data.shape)
finetune_net.params['conv_combine'][1].data.flat=0*np.ones(finetune_net.params['conv_combine'][0].data.shape)
finetune_net.save('../guidedInitModel.caffemodel')


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