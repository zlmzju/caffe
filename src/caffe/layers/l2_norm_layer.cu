#include <vector>

#include "caffe/layers/l2_norm_layer.hpp" 
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void L2NormLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  Dtype* norm_scale = norm_.mutable_cpu_data();
  Dtype normsqr;
  int n = bottom[0]->num();
  int d = bottom[0]->count() / n;
  caffe_gpu_powx(n*d, bottom_data, Dtype(2), top_data);
  for (int i=0; i<n; ++i) {
    caffe_gpu_asum<Dtype>(d, top_data+i*d, &normsqr);
    if (normsqr < eps_) normsqr = eps_;
    norm_scale[i] = pow(normsqr, Dtype(-0.5));
    caffe_gpu_scale<Dtype>(d, norm_scale[i], bottom_data + i*d, top_data + i*d);
  }
}

template <typename Dtype>
void L2NormLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  const Dtype* top_diff = top[0]->gpu_diff();
  const Dtype* top_data = top[0]->gpu_data();
  const Dtype* bottom_data = bottom[0]->gpu_data();
  const Dtype* norm_scale = norm_.cpu_data();
  Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
  const int n = top[0]->num();
  const int d = top[0]->count() / n;
  Dtype a;
  caffe_copy(bottom[0]->count(), top_diff, bottom_diff);
  for (int i=0; i<n; ++i) {
    caffe_gpu_dot(d, top_data+i*d, top_diff+i*d, &a);
    caffe_gpu_axpby(d, Dtype(-1) * a * norm_scale[i], top_data + i*d, norm_scale[i], bottom_diff + i*d);
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(L2NormLayer);
}  // namespace caffe
