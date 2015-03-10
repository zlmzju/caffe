---
name: FCN-32s Fully Convolutional Semantic Segmentation on PASCAL
caffemodel: fcn-32s-pascal.caffemodel
caffemodel_url: http://dl.caffe.berkeleyvision.org/fcn-32s-pascal.caffemodel
sha1: 9ef2dbc22768bd562ba5eec2ad6cae3f499141d0
gist_id: ac410cad48a088710872
---

This is a model from the [paper](http://cs.berkeley.edu/~jonlong/long_shelhamer_fcn.pdf):

    Fully Convolutional Networks for Semantic Segmentation
    Jonathan Long, Evan Shelhamer, Trevor Darrell
    arXiv:1411.4038

This is the single stream, 32 pixel prediction stride version.

This model was trained for the 21-class (including background) PASCAL VOC segmentation task. The final layer outputs scores for each class, which may be normalized via softmax or argmaxed to obtain per-pixel labels. The first label (index zero) is background, with the rest following the standard alphabetical ordering.

The input is expected in BGR channel order, with the following per-channel mean subtracted:

    B 104.00698793 G 116.66876762 R 122.67891434

This is a pre-release: it requires unmerged PRs to run. It should be usable with the branch available at https://github.com/longjon/caffe/tree/future. Training ought to be possible with that code, but the original training scripts have not yet been ported.

This model obtains 64.0 mean I/U on PASCAL seg val 2011.
