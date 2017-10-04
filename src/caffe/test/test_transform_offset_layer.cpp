#include <vector>
#include <iostream>
#include <cmath>
using namespace std;

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layers/transform_offset_layer.hpp"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

namespace caffe {
// Reference convolution for checking results:
// accumulate through explicit loops over input, output, and filters.
template <typename Dtype>
void transform_offset(const Blob<Dtype>* in, TransformOffsetParameter* transform_param,
    Blob<Dtype>* out) {
  // Kernel size, stride, and pad
  int kernel_h, kernel_w;
  kernel_h = kernel_w = sqrt(transform_param->num_output()/2);
  // TransformOffset
  Dtype* out_data = out->mutable_cpu_data();
  for (int n = 0; n < out->shape(0); n++) {
    for (int h = 0; h < out->shape(2); h++) {
      for (int w = 0; w < out->shape(3); w++) {
        Dtype T[8];
        for (int c = 0; c < 8; c++) {
          T[c] = in->data_at(n, c, h, w);
        }
        for (int i = 0; i < kernel_h; i++) {
          for (int j = 0; j < kernel_w; j++) {
            Dtype sx = (kernel_h - 1.0) / 2.0;
            Dtype sy = (kernel_w - 1.0) / 2.0;
            Dtype x = (i - sx);
            Dtype y = (j - sy);
            Dtype x_new = (T[0] + 1.0) * x + T[1] * y + T[2];
            Dtype y_new = T[3] * x + (T[4] + 1.0) * y + T[5];
            Dtype z_new = T[6] * x + T[7] * y + 1.0;
            z_new = 1.0;
            out_data[out->offset(n, 2 * (i * kernel_w + j) + 0, h, w)] = (x_new / z_new - x);
            out_data[out->offset(n, 2 * (i * kernel_w + j) + 1, h, w)] = (y_new / z_new - y);
          }
        }
      }
    }
  }
}

template void transform_offset(const Blob<float>* in,
    TransformOffsetParameter* transform_param,
    Blob<float>* out);
template void transform_offset(const Blob<double>* in,
    TransformOffsetParameter* transform_param,
    Blob<double>* out);


template <typename TypeParam>
class TransformOffsetLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  TransformOffsetLayerTest()
      : blob_bottom_(new Blob<Dtype>(2, 8, 5, 6)),
        blob_top_(new Blob<Dtype>()) {}
  virtual void SetUp() {
    // fill the values
    FillerParameter filler_param;
    GaussianFiller<Dtype> filler(filler_param);
    filler.Fill(blob_bottom_);
    blob_bottom_vec_.push_back(blob_bottom_);
    blob_top_vec_.push_back(blob_top_);
  }

  virtual ~TransformOffsetLayerTest() {
    delete blob_bottom_;
    delete blob_top_;
  }

  virtual Blob<Dtype>* MakeReferenceTop(Blob<Dtype>* top) {
    this->ref_blob_top_.reset(new Blob<Dtype>());
    this->ref_blob_top_->ReshapeLike(*top);
    return this->ref_blob_top_.get();
  }

  Blob<Dtype>* const blob_bottom_;
  Blob<Dtype>* const blob_top_;
  shared_ptr<Blob<Dtype> > ref_blob_top_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

typedef ::testing::Types<GPUDevice<float>, GPUDevice<double> > TestDtypesGPU;
TYPED_TEST_CASE(TransformOffsetLayerTest, TestDtypesGPU);

TYPED_TEST(TransformOffsetLayerTest, TestSimpleTransformOffset) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  TransformOffsetParameter* transform_param =
      layer_param.mutable_transform_offset_param();
  transform_param->set_num_output(98); 
  shared_ptr<Layer<Dtype> > layer(
      new TransformOffsetLayer<Dtype>(layer_param));
  layer->Reshape(this->blob_bottom_vec_, this->blob_top_vec_);
  layer->Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  // Check against reference convolution.
  const Dtype* top_data;
  const Dtype* ref_top_data;
  transform_offset(this->blob_bottom_, transform_param, this->MakeReferenceTop(this->blob_top_));
  top_data = this->blob_top_->cpu_data();
  ref_top_data = this->ref_blob_top_->cpu_data();
  for (int i = 0; i < this->blob_top_->count(); ++i) {
    EXPECT_NEAR(top_data[i], ref_top_data[i], 1e-4);
  }
}
//Gradient
TYPED_TEST(TransformOffsetLayerTest, TestGradient) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  TransformOffsetParameter* transform_param =
      layer_param.mutable_transform_offset_param();
  transform_param->set_num_output(50);
  TransformOffsetLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_,this->blob_top_vec_);
  GradientChecker<Dtype> checker(1e-2, 1e-3);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_, this->blob_top_vec_);
}

}  // namespace caffe
