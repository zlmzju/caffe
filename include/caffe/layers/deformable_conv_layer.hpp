#ifndef CAFFE_DEFORMABEL_CONV_LAYER_HPP_
#define CAFFE_DEFORMABLE_CONV_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/deformable_im2col.hpp"

#include "caffe/layers/base_conv_layer.hpp"

namespace caffe {

template <typename Dtype>
class DeformableConvolutionLayer : public BaseConvolutionLayer<Dtype> {
 public:
  explicit DeformableConvolutionLayer(const LayerParameter& param)
      : BaseConvolutionLayer<Dtype>(param) {}

  virtual inline const char* type() const { return "DeformableConvolution"; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual inline bool reverse_dimensions() { return false; }
  virtual void compute_output_shape();
};

}  // namespace caffe

#endif  // CAFFE_DEFORMABLE__CONV_LAYER_HPP_
