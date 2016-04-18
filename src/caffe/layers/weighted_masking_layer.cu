#include <vector>

#include "caffe/layers/weighted_masking_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void WeightedMaskingLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  const Dtype* masking_data = bottom[1]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  const int size = bottom[0]->count(2);

  for (int n = 0; n < bottom[0]->num(); ++n) {
    for (int c = 0; c < bottom[0]->channels(); ++c) {
      caffe_gpu_mul(size, bottom_data, masking_data, top_data);
      bottom_data += size;
      top_data += size;
    }
    masking_data += size;
  }
}

template <typename Dtype>
void WeightedMaskingLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  const int size = bottom[0]->count(2);
  if (propagate_down[0]) {
    Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
    const Dtype* top_diff = top[0]->gpu_diff();
    const Dtype* masking_data = bottom[1]->gpu_data();
    for (int n = 0; n < bottom[0]->num(); ++n) {
      for (int c = 0; c < bottom[0]->channels(); ++c) {
        caffe_gpu_mul(size, top_diff, masking_data, bottom_diff);
        bottom_diff += size;
        top_diff += size;
      }
      masking_data += size;
    }
  }
  if (propagate_down[1]) {
    Dtype* masking_diff = bottom[1]->mutable_gpu_diff();
    const Dtype* top_diff = top[0]->gpu_diff();
    const Dtype* bottom_data = bottom[0]->gpu_data();
    for (int n = 0; n < bottom[0]->num(); ++n) {
      caffe_gpu_mul(size, top_diff, bottom_data, masking_diff);
      bottom_data += size;
      top_diff += size;
      for (int c = 1; c < bottom[0]->channels(); ++c) {
        caffe_gpu_mul(size, top_diff, bottom_data, buffer_.mutable_gpu_data());
        caffe_gpu_axpy(size, Dtype(1.), buffer_.gpu_data(), masking_diff);
        bottom_data += size;
        top_diff += size;
      }
      masking_diff += size;
    }
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(WeightedMaskingLayer);
}  // namespace caffe