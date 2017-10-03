#include <vector>
#include <cmath>

#include "caffe/layers/transform_offset_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void TransformOffsetLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  // Figure out the dimensions
  const int C = this->layer_param_.transform_offset_param().num_output();
  vector<int> top_shape = bottom[0]->shape();
  top_shape[1] = C;
  top[0]->Reshape(top_shape);
  this->kernel_size = (int) sqrt(C/2);
}
#ifdef CPU_ONLY
STUB_GPU(TransformOffsetLayer);
#endif

INSTANTIATE_CLASS(TransformOffsetLayer);
REGISTER_LAYER_CLASS(TransformOffset);

}  // namespace caffe
