#include <vector>
#include <iostream>

#include "caffe/layers/deformable_conv_layer.hpp"
#include "caffe/util/deformable_im2col.hpp"
using namespace std;


namespace caffe {
template <typename Dtype>
void DeformableConvolutionLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
    const Dtype* weights = this->blobs_[0]->gpu_data();
    const Dtype* bottom_data = bottom[0]->gpu_data();
    const Dtype* offset = bottom[1]->gpu_data();
    Dtype* top_data = top[0]->mutable_gpu_data();
    Dtype* col_buff = col_buffer_.mutable_gpu_data();


    for (int n = 0; n < this->num_; ++n) {
        deformable_im2col_gpu(bottom_data + n*this->bottom_dim_, //data_col
               offset + n*this->offset_dim_,//offset
               this->channels_,
               bottom[0]->shape(2),//height 
               bottom[0]->shape(3),//width
               this->kernel_shape_.cpu_data()[0],//
               this->kernel_shape_.cpu_data()[1],
               this->pad_.cpu_data()[0],
               this->pad_.cpu_data()[1],
               this->stride_.cpu_data()[0],
               this->stride_.cpu_data()[1],
               this->dilation_.cpu_data()[0],
               this->dilation_.cpu_data()[1],
               1,//deformable group
               col_buff);

        for (int g = 0; g < this->group_; ++g) {
            caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, 
                  this->conv_out_channels_ /this->group_, this->conv_out_spatial_dim_, this->kernel_dim_,
                  (Dtype)1., weights + this->weight_offset_ * g, col_buff + this->col_offset_ * g,
                  (Dtype)0., top_data + n * this->top_dim_ + this->output_offset_ * g);
        }
        if (this->bias_term_) {
          const Dtype* bias = this->blobs_[1]->gpu_data();
          caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, this->num_output_,
                this->out_spatial_dim_, 1, (Dtype)1., bias, bias_multiplier_.gpu_data(),
                (Dtype)1., top_data + n * this->top_dim_);
        }
    }
}

template <typename Dtype>
void DeformableConvolutionLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
   const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
    const Dtype* weights = this->blobs_[0]->gpu_data();
    const Dtype* bottom_data = bottom[0]->gpu_data();
    const Dtype* offset = bottom[1]->gpu_data();
    Dtype* weight_diff = this->blobs_[0]->mutable_gpu_diff();

    Dtype* col_buff = col_buffer_.mutable_gpu_data();
    const Dtype* top_diff = top[0]->gpu_diff();
    // Bias gradient, if necessary.
    if (this->bias_term_ && this->param_propagate_down_[1]) {
      Dtype* bias_diff = this->blobs_[1]->mutable_gpu_diff();
      for (int n = 0; n < this->num_; ++n) {
        caffe_gpu_gemv<Dtype>(CblasNoTrans, this->num_output_, this->out_spatial_dim_, 1.,
             top_diff+n*this->top_dim_, bias_multiplier_.gpu_data(), 1., bias_diff);  
      }
    }
    if (this->param_propagate_down_[0] || propagate_down[0]) {
      Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
      Dtype* offset_diff = bottom[1]->mutable_gpu_diff();
      for (int n = 0; n < this->num_; ++n) {
        // gradient w.r.t. weight. Note that we will accumulate diffs.
        if (this->param_propagate_down_[0]) {
            for (int g = 0; g < this->group_; ++g) {
                caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, 
                      this->conv_out_channels_ /this->group_, this->kernel_dim_, this->conv_out_spatial_dim_, 
                      (Dtype)1., top_diff + n * this->top_dim_ + this->output_offset_ * g,
                      col_buffer_.gpu_data() + this->col_offset_ * g,
                      (Dtype)1., weight_diff + this->weight_offset_ * g);
                // gradient w.r.t. bottom data, if necessary.
                caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, this->kernel_dim_, 
                      this->conv_out_spatial_dim_, this->conv_out_channels_ /this->group_,
                      (Dtype)1., weights + this->weight_offset_ * g,
                      top_diff + n * this->top_dim_ + this->output_offset_ * g,
                      (Dtype)0., col_buff + this->col_offset_ * g);
           }


           cout<<"grad. data"<<endl;
           // gradient w.r.t input data
           deformable_col2im_gpu(col_buffer_.gpu_data(), //data_col
                 offset + n*this->bottom_dim_,//offset
                 this->channels_,
                 bottom[0]->shape(2),//height 
                 bottom[0]->shape(3),//width
                 this->kernel_shape_.cpu_data()[0],//kernel_h
                 this->kernel_shape_.cpu_data()[1],//kernel_w
                 this->pad_.cpu_data()[0],
                 this->pad_.cpu_data()[1],
                 this->stride_.cpu_data()[0],
                 this->stride_.cpu_data()[1],
                 this->dilation_.cpu_data()[0],
                 this->dilation_.cpu_data()[1],
                 1,//deformable group
                 bottom_diff + n*this->bottom_dim_);

           cout<<"grad. offset"<<endl;
           // gradient w.r.t input offset data
           deformable_col2im_coord_gpu(col_buffer_.gpu_data(), //data_col
                 offset + n*this->offset_dim_,//offset
                 this->channels_,
                 bottom[0]->shape(2),//height 
                 bottom[0]->shape(3),//width
                 this->kernel_shape_.cpu_data()[0],//kernel_h
                 this->kernel_shape_.cpu_data()[1],//kernel_w
                 this->pad_.cpu_data()[0],
                 this->pad_.cpu_data()[1],
                 this->stride_.cpu_data()[0],
                 this->stride_.cpu_data()[1],
                 this->dilation_.cpu_data()[0],
                 this->dilation_.cpu_data()[1],
                 1,//deformable group
                 offset_diff + n*this->offset_dim_);
        }
      }

    }
}
//

INSTANTIATE_LAYER_GPU_FUNCS(DeformableConvolutionLayer);

}  // namespace caffe
