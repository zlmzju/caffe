#include <vector>
#include <iostream>
using namespace std;

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layers/deformable_conv_layer.hpp"

#ifdef USE_CUDNN
#include "caffe/layers/cudnn_conv_layer.hpp"
#endif

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

namespace caffe {

template <typename Dtype>
void debug_info(Blob<Dtype> *blob, bool diff=false){
    const Dtype* array=blob->cpu_data();
    if(diff) array=blob->cpu_diff();
    int C=blob->shape(1);
    int L=blob->shape(2);
    for(int n=0;n<blob->shape(0);n++){
     for(int c=0;c<C;c++){
      for(int i=0;i<L;i++){
       for(int j=0;j<L;j++){
         cout<<array[c*L*L+i*L+j]<<",";
       }
       cout<<endl;
      }
      cout<<endl;
     }
    }
}


template <typename Dtype>
Dtype get_data(const Blob<Dtype>* in, int n, int c, Dtype h, Dtype w) {
  int h_low = floor(h);
  int w_low = floor(w);
  int h_high;
  int w_high;
  const int height = in->shape(2);
  const int width = in->shape(3);
  if (h_low >= height - 1) {
	h_high = h_low = height - 1;
	h = (Dtype)h_low;
  }
  else {
	h_high = h_low + 1;
  }

  if (w_low >= width - 1) {
	w_high = w_low = width - 1;
	w = (Dtype)w_low;
  }
  else {
	w_high = w_low + 1;
  }

  Dtype lh = h - h_low;
  Dtype lw = w - w_low;
  Dtype hh = 1 - lh, hw = 1 - lw;

  Dtype v1 = in->data_at(n,c,h_low,w_low);//
  Dtype v2 = in->data_at(n,c,h_low,w_high);//ottom_data[h_low * data_width + w_high];
  Dtype v3 = in->data_at(n,c,h_high,w_low);//ottom_data[h_high * data_width + w_low];
  Dtype v4 = in->data_at(n,c,h_high,w_high);//ottom_data[h_high * data_width + w_high];
  Dtype w1 = hh * hw, w2 = hh * lw, w3 = lh * hw, w4 = lh * lw;

  Dtype val = (w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4);
  return val;


}
// Reference convolution for checking results:
// accumulate through explicit loops over input, output, and filters.
template <typename Dtype>
void caffe_conv(const Blob<Dtype>* in, const Blob<Dtype>* in_off, ConvolutionParameter* conv_param,
    const vector<shared_ptr<Blob<Dtype> > >& weights,
    Blob<Dtype>* out) {
  const bool has_depth = (out->num_axes() == 5);
  if (!has_depth) { CHECK_EQ(4, out->num_axes()); }
  // Kernel size, stride, and pad
  int kernel_h, kernel_w;
  if (conv_param->has_kernel_h() || conv_param->has_kernel_w()) {
    kernel_h = conv_param->kernel_h();
    kernel_w = conv_param->kernel_w();
  } else {
    kernel_h = kernel_w = conv_param->kernel_size(0);
  }
  int pad_h, pad_w;
  if (conv_param->has_pad_h() || conv_param->has_pad_w()) {
    pad_h = conv_param->pad_h();
    pad_w = conv_param->pad_w();
  } else {
    pad_h = pad_w = conv_param->pad_size() ? conv_param->pad(0) : 0;
  }
  int stride_h, stride_w;
  if (conv_param->has_stride_h() || conv_param->has_stride_w()) {
    stride_h = conv_param->stride_h();
    stride_w = conv_param->stride_w();
  } else {
    stride_h = stride_w = conv_param->stride_size() ? conv_param->stride(0) : 1;
  }
  int dilation_h, dilation_w;
  dilation_h = dilation_w = conv_param->dilation_size() ?
                            conv_param->dilation(0) : 1;
  int kernel_d, pad_d, stride_d, dilation_d;
  if (has_depth) {
    kernel_d = kernel_h;
    stride_d = stride_h;
    pad_d = pad_h;
    dilation_d = dilation_h;
  } else {
    kernel_d = stride_d = dilation_d = 1;
    pad_d = 0;
  }
  // Groups
  int groups = conv_param->group();
  int o_g = out->shape(1) / groups;
  int k_g = in->shape(1) / groups;
  int o_head, k_head;
  // Convolution
  vector<int> weight_offset(4 + has_depth);
  vector<int> in_offset(4 + has_depth);
  vector<int> out_offset(4 + has_depth);
  Dtype* out_data = out->mutable_cpu_data();
  for (int n = 0; n < out->shape(0); n++) {
    for (int g = 0; g < groups; g++) {
      o_head = o_g * g;
      k_head = k_g * g;
      for (int o = 0; o < o_g; o++) {
        for (int k = 0; k < k_g; k++) {
          for (int z = 0; z < (has_depth ? out->shape(2) : 1); z++) {
            for (int y = 0; y < out->shape(2 + has_depth); y++) {
              for (int x = 0; x < out->shape(3 + has_depth); x++) {
                for (int r = 0; r < kernel_d; r++) {
                  for (int p = 0; p < kernel_h; p++) {
                    for (int q = 0; q < kernel_w; q++) {
                      int in_z = z * stride_d - pad_d + r * dilation_d;
                      int in_y = y * stride_h - pad_h + p * dilation_h;
                      int in_x = x * stride_w - pad_w + q * dilation_w;
                      Dtype dy = in_off->data_at(n, (p*kernel_w+q)*2 + 0, y, x);
                      Dtype dx = in_off->data_at(n, (p*kernel_w+q)*2 + 1, y, x);
                      if (in_z >= 0 && in_z < (has_depth ? in->shape(2) : 1)) {
                        weight_offset[0] = o + o_head;
                        weight_offset[1] = k;
                        if (has_depth) { weight_offset[2] = r; }
                        weight_offset[2 + has_depth] = p;
                        weight_offset[3 + has_depth] = q;
                        in_offset[0] = n;
                        in_offset[1] = k + k_head;
                        if (has_depth) { in_offset[2] = in_z; }
                        in_offset[2 + has_depth] = in_y;
                        in_offset[3 + has_depth] = in_x;
                        out_offset[0] = n;
                        out_offset[1] = o + o_head;
                        if (has_depth) { out_offset[2] = z; }
                        out_offset[2 + has_depth] = y;
                        out_offset[3 + has_depth] = x;
                        Dtype val_in = 0.0;
                        Dtype ry = in_y + dy;
                        Dtype rx = in_x + dx;
                        if( ry >= 0 && rx >=0 && ry < in->shape(2) && rx < in->shape(3))
                          val_in = get_data(in, in_offset[0], in_offset[1], in_y + dy, in_x + dx);
                        out_data[out->offset(out_offset)] +=
                            val_in
                            * weights[0]->data_at(weight_offset);
                      }
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
  }
  // Bias
  if (conv_param->bias_term()) {
    const Dtype* bias_data = weights[1]->cpu_data();
    for (int n = 0; n < out->shape(0); n++) {
      for (int o = 0; o < out->shape(1); o++) {
        for (int z = 0; z < (has_depth ? out->shape(2) : 1); z++) {
          for (int y = 0; y < out->shape(2 + has_depth); y++) {
            for (int x = 0; x < out->shape(3 + has_depth); x++) {
              out_offset[0] = n;
              out_offset[1] = o;
              if (has_depth) { out_offset[2] = z; }
              out_offset[2 + has_depth] = y;
              out_offset[3 + has_depth] = x;
              out_data[out->offset(out_offset)] += bias_data[o];
            }
          }
        }
      }
    }
  }
}

template void caffe_conv(const Blob<float>* in, const Blob<float>* in_off,
    ConvolutionParameter* conv_param,
    const vector<shared_ptr<Blob<float> > >& weights,
    Blob<float>* out);
template void caffe_conv(const Blob<double>* in, const Blob<double>* in_off,
    ConvolutionParameter* conv_param,
    const vector<shared_ptr<Blob<double> > >& weights,
    Blob<double>* out);

template <typename TypeParam>
class DeformableConvolutionLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  DeformableConvolutionLayerTest()
      : blob_bottom_(new Blob<Dtype>(1, 1, 4, 4)),
        blob_bottom_2_(new Blob<Dtype>(1, 18, 4, 4)),
        blob_top_(new Blob<Dtype>()) {}
  virtual void SetUp() {
    // fill the values
    FillerParameter filler_param;
    filler_param.set_value(1);
    GaussianFiller<Dtype> filler(filler_param);
    filler.Fill(blob_bottom_);
    FillerParameter filler_param2;
    filler_param2.set_value(1.0);
    ConstantFiller<Dtype> filler2(filler_param2);
    filler2.Fill(blob_bottom_2_);
    blob_bottom_vec_.push_back(blob_bottom_);
    blob_bottom_vec_.push_back(blob_bottom_2_);
    blob_top_vec_.push_back(blob_top_);
  }

  virtual ~DeformableConvolutionLayerTest() {
    delete blob_bottom_;
    delete blob_bottom_2_;
    delete blob_top_;
  }

  virtual Blob<Dtype>* MakeReferenceTop(Blob<Dtype>* top) {
    this->ref_blob_top_.reset(new Blob<Dtype>());
    this->ref_blob_top_->ReshapeLike(*top);
    return this->ref_blob_top_.get();
  }

  Blob<Dtype>* const blob_bottom_;
  Blob<Dtype>* const blob_bottom_2_;
  Blob<Dtype>* const blob_top_;
  shared_ptr<Blob<Dtype> > ref_blob_top_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(DeformableConvolutionLayerTest, TestDtypesAndDevices);

TYPED_TEST(DeformableConvolutionLayerTest, TestSimpleConvolution) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  ConvolutionParameter* convolution_param =
      layer_param.mutable_convolution_param();
  convolution_param->add_kernel_size(3);
  convolution_param->add_stride(1);
  convolution_param->add_pad(1);
  convolution_param->set_num_output(1);
  convolution_param->mutable_weight_filler()->set_type("constant");
  convolution_param->mutable_weight_filler()->set_value(1.0);
  convolution_param->mutable_bias_filler()->set_type("constant");
  convolution_param->mutable_bias_filler()->set_value(1.0);
  shared_ptr<Layer<Dtype> > layer(
      new DeformableConvolutionLayer<Dtype>(layer_param));
  layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer->Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  // Check against reference convolution.
  const Dtype* top_data;
  const Dtype* ref_top_data;
  caffe_conv(this->blob_bottom_, this->blob_bottom_2_, convolution_param, layer->blobs(),
      this->MakeReferenceTop(this->blob_top_));
  top_data = this->blob_top_->cpu_data();
  ref_top_data = this->ref_blob_top_->cpu_data();
  for (int i = 0; i < this->blob_top_->count(); ++i) {
    EXPECT_NEAR(top_data[i], ref_top_data[i], 1e-4);
  }
  debug_info(this->blob_bottom_);
}
//Gradient
TYPED_TEST(DeformableConvolutionLayerTest, TestDataGradient) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  ConvolutionParameter* convolution_param =
      layer_param.mutable_convolution_param();
  convolution_param->add_kernel_size(3);
  convolution_param->add_stride(1);
  convolution_param->add_pad(1);
  convolution_param->set_num_output(1);
  convolution_param->mutable_weight_filler()->set_type("gaussian");
  convolution_param->mutable_bias_filler()->set_type("constant");
  convolution_param->mutable_bias_filler()->set_value(0.0);

  FillerParameter filler_param2;
  filler_param2.set_value(1.);
  GaussianFiller<Dtype> filler2(filler_param2);
  filler2.Fill(this->blob_bottom_2_);
  DeformableConvolutionLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_,this->blob_top_vec_);
  GradientChecker<Dtype> checker(1e-2, 1e-3);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_, this->blob_top_vec_, 0);
//  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_, this->blob_top_vec_, 1);
}

TYPED_TEST(DeformableConvolutionLayerTest, TestOffsetGradient) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  ConvolutionParameter* convolution_param =
      layer_param.mutable_convolution_param();
  convolution_param->add_kernel_size(3);
  convolution_param->add_stride(1);
  convolution_param->add_pad(1);
  convolution_param->set_num_output(1);
  convolution_param->mutable_weight_filler()->set_type("constant");
  convolution_param->mutable_weight_filler()->set_value(1);
  convolution_param->mutable_bias_filler()->set_type("constant");
  convolution_param->mutable_bias_filler()->set_value(0);
  FillerParameter filler_param2;
  filler_param2.set_value(1.);
  ConstantFiller<Dtype> filler2(filler_param2);
  filler2.Fill(this->blob_bottom_2_);
  DeformableConvolutionLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_,this->blob_top_vec_);
  GradientChecker<Dtype> checker(1e-1, 1e-3);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_, this->blob_top_vec_, 1);
//  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_, this->blob_top_vec_, 1);
}
}  // namespace caffe
