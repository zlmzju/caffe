#include <vector>

#include "caffe/layers/weighted_masking_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void WeightedMaskingLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  buffer_.Reshape(1, 1, bottom[0]->height(), bottom[0]->width());
}

template <typename Dtype>
void WeightedMaskingLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  top[0]->ReshapeLike(*bottom[0]);
}

template <typename Dtype>
void WeightedMaskingLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  const Dtype* masking_data = bottom[1]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  const int size = bottom[0]->count(2);

  for (int n = 0; n < bottom[0]->num(); ++n) {
    for (int c = 0; c < bottom[0]->channels(); ++c) {
      caffe_mul(size, bottom_data, masking_data, top_data);
      bottom_data += size;
      top_data += size;
    }
    masking_data += size;
  }
}

template <typename Dtype>
void WeightedMaskingLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  const int size = bottom[0]->height()*bottom[0]->width();
  if (propagate_down[0]) {
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    const Dtype* top_diff = top[0]->cpu_diff();
    const Dtype* masking_data = bottom[1]->cpu_data();
    for (int n = 0; n < bottom[0]->num(); ++n) {
      for (int c = 0; c < bottom[0]->channels(); ++c) {
        caffe_mul(size, top_diff, masking_data, bottom_diff);
        bottom_diff += size;
        top_diff += size;
      }
      masking_data += size;
    }
  }
  if (propagate_down[1]) {
    Dtype* masking_diff = bottom[1]->mutable_cpu_diff();
    const Dtype* top_diff = top[0]->cpu_diff();
    const Dtype* bottom_data = bottom[0]->cpu_data();
    for (int n = 0; n < bottom[0]->num(); ++n) {
      caffe_mul(size, top_diff, bottom_data, masking_diff);
      bottom_data += size;
      top_diff += size;
      for (int c = 1; c < bottom[0]->channels(); ++c) {
        caffe_mul(size, top_diff, bottom_data, buffer_.mutable_cpu_data());
        caffe_axpy(size, Dtype(1.), buffer_.cpu_data(), masking_diff);
        bottom_data += size;
        top_diff += size;
      }
      masking_diff +=size;
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU(WeightedMaskingLayer);
#endif

INSTANTIATE_CLASS(WeightedMaskingLayer);
REGISTER_LAYER_CLASS(WeightedMasking);

}  // namespace caffe