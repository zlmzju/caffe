#include <vector>
#include <cmath>

#include "caffe/layers/transform_anchor_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void TransformAnchorLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  // Figure out the dimensions
  this->stride = (Dtype) this->layer_param_.transform_anchor_param().stride();
  this->kernel_size = this->layer_param_.transform_anchor_param().kernel_size();
  vector<int> top_shape = bottom[0]->shape();
  top_shape[1] = 8;
  top[0]->Reshape(top_shape);
}
#ifdef CPU_ONLY
STUB_GPU(TransformAnchorLayer);
#endif

INSTANTIATE_CLASS(TransformAnchorLayer);
REGISTER_LAYER_CLASS(TransformAnchor);

}  // namespace caffe
