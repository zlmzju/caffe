#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
import caffe
def upsample_filt(size):
    factor = (size + 1) // 2    #integer division
    if size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:size, :size]
    return (1 - abs(og[0] - center) / factor) * \
           (1 - abs(og[1] - center) / factor)
# Make sure that caffe is on the python path:
caffe_root = '/home/liming/project/caffe/'  # this file is expected to be in {caffe_root}/examples
# Set the right path to your model definition file, pretrained model weights,
# and the image you would like to classify.
MODEL_FILE = 'VGG_ILSVRC_16_layers_deploy.prototxt'
PRETRAINED = 'VGG_ILSVRC_16_layers.caffemodel'
caffe.set_mode_gpu()
net = caffe.Classifier(MODEL_FILE, PRETRAINED,caffe.TEST)
print('\nblobs')
print [(k, v.data.shape) for k, v in net.blobs.items()]
print('\nparams')
print [(k, v[0].data.shape) for k, v in net.params.items()]

NEWMODEL_FILE = 'vgg16fc_deploy.prototxt'
NEWPRETRAINED = 'VGG_ILSVRC_16_layers.caffemodel'
finetune_net = caffe.Classifier(NEWMODEL_FILE, NEWPRETRAINED,
                       caffe.TEST)
print('\nparams')
print [(k, v[0].data.shape) for k, v in finetune_net.params.items()]
finetune_net.params['conv6'][0].data.flat=net.params['fc6'][0].data.flat
finetune_net.params['conv6'][1].data[...]=net.params['fc6'][1].data
finetune_net.params['conv6'][0].data.flat=net.params['fc6'][0].data.flat
finetune_net.params['conv7'][1].data[...]=net.params['fc7'][1].data
#finetune_net.params['score'][0].data.flat
finetune_net.params['deconv'][0].data.flat=upsample_filt(finetune_net.params['deconv'][0].data.shape[3])
#finetune_net.save('vgg16fc.caffemodel')

#test the net
transformer = caffe.io.Transformer({'data': finetune_net.blobs['data'].data.shape})
transformer.set_transpose('data', (2,0,1))
transformer.set_mean('data', np.load(caffe_root + 'python/caffe/imagenet/ilsvrc_2012_mean.npy').mean(1).mean(1)) # mean pixel
transformer.set_raw_scale('data', 255)  # the reference model operates on images in [0,255] range instead of [0,1]
transformer.set_channel_swap('data', (2,1,0))  # the reference model has channels in BGR order instead of RGB
finetune_net.blobs['data'].reshape(1,3,500,500)
finetune_net.blobs['data'].data[...] = transformer.preprocess('data', caffe.io.load_image(caffe_root+'examples/images/cat.jpg'))

out=finetune_net.forward()
outmap=np.squeeze(out['outmap'])
plt.imshow(outmap)