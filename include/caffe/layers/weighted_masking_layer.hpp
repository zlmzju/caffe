#ifndef CAFFE_WEIGHTED_MASKING_HPP_
#define CAFFE_WEIGHTED_MASKING_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

/**
 * @brief Given feature map bottom[0] (N,C,H,W) and weighting map bottom[1] (N,1,H,W) 
 *        output weighted top (N,C,H,W) by elementwise product each channel of 
 *        the feature map by the weighting map.
 */
template <typename Dtype>
class WeightedMaskingLayer : public Layer<Dtype> {
 public:
  explicit WeightedMaskingLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}

  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual inline const char* type() const { return "WeightedMasking"; }

  virtual inline int ExactNumBottomBlobs() const { return 2; }
  virtual inline int ExactNumTopBlobs() const { return 1; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  Blob<Dtype> buffer_;  // temporary buffer for masking_diff accumulation
};

}  // namespace caffe

#endif  // CAFFE_WEIGHTED_MASKING_HPP_