#include <vector>
#include "caffe/filler.hpp"
#include <iostream>
#include "caffe/layers/deformable_conv_layer.hpp"
using namespace std;
namespace caffe {
template <typename Dtype>
 int DeformableConvolutionLayer<Dtype>::input_shape(int i) {
   return (*(this->bottom_shape_))[this->channel_axis_ + i];
  }


template <typename Dtype>
void DeformableConvolutionLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const int first_spatial_axis = this->channel_axis_ + 1;
  CHECK_EQ(bottom[0]->num_axes(), first_spatial_axis + this->num_spatial_axes_)
      << "bottom num_axes may not change.";
  this->num_ = bottom[0]->count(0, this->channel_axis_);
  CHECK_EQ(bottom[0]->shape(this->channel_axis_), this->channels_)
      << "Input size incompatible with convolution kernel.";
  // TODO: generalize to handle inputs of different shapes.
  // Shape the tops.
  this->bottom_shape_ = &bottom[0]->shape();
  compute_output_shape();
  vector<int> top_shape(bottom[0]->shape().begin(),
      bottom[0]->shape().begin() + this->channel_axis_);
  top_shape.push_back(this->num_output_);
  for (int i = 0; i < this->num_spatial_axes_; ++i) {
    top_shape.push_back(this->output_shape_[i]);
  }
  for (int top_id = 0; top_id < top.size(); ++top_id) {
    top[top_id]->Reshape(top_shape);

  }
  
  if (reverse_dimensions()) {
    this->conv_out_spatial_dim_ = bottom[0]->count(first_spatial_axis);
  } else {
    this->conv_out_spatial_dim_ = top[0]->count(first_spatial_axis);
  }
  this->col_offset_ = this->kernel_dim_ * this->conv_out_spatial_dim_;
  this->output_offset_ = this->conv_out_channels_ * this->conv_out_spatial_dim_ / this->group_;
  // Setup input dimensions (conv_input_shape_).
  vector<int> bottom_dim_blob_shape(1, this->num_spatial_axes_ + 1);
  this->conv_input_shape_.Reshape(bottom_dim_blob_shape);
  int* conv_input_shape_data = this->conv_input_shape_.mutable_cpu_data();
  for (int i = 0; i < this->num_spatial_axes_ + 1; ++i) {
    if (reverse_dimensions()) {
      conv_input_shape_data[i] = top[0]->shape(this->channel_axis_ + i);
    } else {
      conv_input_shape_data[i] = bottom[0]->shape(this->channel_axis_ + i);
    }
  }
  // The im2col result buffer will only hold one image at a time to avoid
  // overly large memory usage. In the special case of 1x1 convolution
  // it goes lazily unused to save memory.
  this->col_buffer_shape_.clear();
  this->col_buffer_shape_.push_back(this->kernel_dim_ * this->group_);
  for (int i = 0; i < this->num_spatial_axes_; ++i) {
    if (reverse_dimensions()) {
      this->col_buffer_shape_.push_back(input_shape(i + 1));
    } else {
      this->col_buffer_shape_.push_back(this->output_shape_[i]);
    }
  }
  this->col_buffer_.Reshape(this->col_buffer_shape_);
  this->bottom_dim_ = bottom[0]->count(this->channel_axis_);
  this->offset_dim_ = bottom[1]->count(this->channel_axis_);
  this->top_dim_ = top[0]->count(this->channel_axis_);
  this->num_kernels_im2col_ = this->conv_in_channels_ * this->conv_out_spatial_dim_;
  this->num_kernels_col2im_ = reverse_dimensions() ? this->top_dim_ : this->bottom_dim_;
  // Set up the all ones "bias multiplier" for adding biases by BLAS
  this->out_spatial_dim_ = top[0]->count(first_spatial_axis);
  if (this->bias_term_) {
    vector<int> bias_multiplier_shape(1, this->out_spatial_dim_);
    this->bias_multiplier_.Reshape(bias_multiplier_shape);
    caffe_set(this->bias_multiplier_.count(), Dtype(1),
        this->bias_multiplier_.mutable_cpu_data());
  }
}





template <typename Dtype>
void DeformableConvolutionLayer<Dtype>::compute_output_shape() {
  const int* kernel_shape_data = this->kernel_shape_.cpu_data();
  const int* stride_data = this->stride_.cpu_data();
  const int* pad_data = this->pad_.cpu_data();
  const int* dilation_data = this->dilation_.cpu_data();
  this->output_shape_.clear();
  for (int i = 0; i < this->num_spatial_axes_; ++i) {
    // i + 1 to skip channel axis
    const int input_dim = this->input_shape(i + 1);
    const int kernel_extent = dilation_data[i] * (kernel_shape_data[i] - 1) + 1;
    const int output_dim = (input_dim + 2 * pad_data[i] - kernel_extent)
        / stride_data[i] + 1;
    this->output_shape_.push_back(output_dim);
  }
}

#ifdef CPU_ONLY
STUB_GPU(DeformableConvolutionLayer);
#endif

INSTANTIATE_CLASS(DeformableConvolutionLayer);
REGISTER_LAYER_CLASS(DeformableConvolution);

}  // namespace caffe
