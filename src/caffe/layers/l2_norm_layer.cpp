#include <algorithm>
#include <vector>
#include <cmath>

#include "caffe/layers/l2_norm_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void L2NormLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  top[0]->Reshape(bottom[0]->num(), bottom[0]->channels(),
                  bottom[0]->height(), bottom[0]->width());
  norm_.Reshape(bottom[0]->num(), 1, 1, 1);
  eps_=Dtype(1e-12);
}

template <typename Dtype>
void L2NormLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  Dtype* norm_scale = norm_.mutable_cpu_data();
  int n = bottom[0]->num();
  int d = bottom[0]->count() / n;
  caffe_sqr<Dtype>(n*d, bottom_data, top_data);
  for (int i=0; i<n; ++i) {
    Dtype normsqr = caffe_cpu_asum<Dtype>(d, top_data + i*d);
    if (normsqr < eps_) normsqr = eps_;
    norm_scale[i] = pow(normsqr, Dtype(-0.5));
    caffe_cpu_scale<Dtype>(d, norm_scale[i], bottom_data + i*d, top_data + i*d);
  }
}

template <typename Dtype>
void L2NormLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  const Dtype* top_diff = top[0]->cpu_diff();
  const Dtype* top_data = top[0]->cpu_data();
  const Dtype* bottom_data = bottom[0]->cpu_data();
  const Dtype* norm_scale = norm_.cpu_data();
  Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
  const int n = top[0]->num();
  const int d = top[0]->count() / n;
  caffe_copy(bottom[0]->count(), top_diff, bottom_diff);
  for (int i=0; i<n; ++i) {
    Dtype a = caffe_cpu_dot(d, top_data+i*d, top_diff+i*d);
    caffe_cpu_axpby(d, Dtype(-1) * a * norm_scale[i], top_data + i*d, norm_scale[i], bottom_diff + i*d);
  }
}


#ifdef CPU_ONLY
STUB_GPU(L2NormLayer);
#endif

INSTANTIATE_CLASS(L2NormLayer);
REGISTER_LAYER_CLASS(L2Norm);

}  // namespace caffe