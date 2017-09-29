#include <vector>
#include <cmath>

#include "caffe/layers/transform_rs_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void TransformRSLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  // Figure out the dimensions
  const int C = this->layer_param_.transform_offset_param().num_output();
  const int K = this->layer_param_.transform_offset_param().kernel_size();
  vector<int> top_shape = bottom[0]->shape();
  top_shape[1] = C;
  top[0]->Reshape(top_shape);
  this->kernel_size = K > 0 ? K: (int) sqrt(C/2);
}
#ifdef CPU_ONLY
STUB_GPU(TransformRSLayer);
#endif

INSTANTIATE_CLASS(TransformRSLayer);
REGISTER_LAYER_CLASS(TransformRS);

}  // namespace caffe
