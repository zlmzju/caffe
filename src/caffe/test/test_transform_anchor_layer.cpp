#include <vector>
#include <iostream>
#include <cmath>
using namespace std;

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layers/transform_anchor_layer.hpp"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

namespace caffe {
// Reference convolution for checking results:
// accumulate through explicit loops over input, output, and filters.
template <typename Dtype>
void transform_anchor(const Blob<Dtype>* in, 
    Blob<Dtype>* out) {
  // Kernel size, stride, and pad
  int kernel_h, kernel_w;
  kernel_h = kernel_w = 7;
  // TransformAnchor
  Dtype* out_data = out->mutable_cpu_data();
  for (int n = 0; n < out->shape(0); n++) {
    for (int h = 0; h < out->shape(2); h++) {
      for (int w = 0; w < out->shape(3); w++) {
        Dtype T[8];
        for (int c = 0; c < 8; c++) {
          T[c] = in->data_at(n, c, h, w);
        }
        int c = 0;
        Dtype box[8] = {-1, -1, 1, -1, 1, 1, -1, 1};
        for (int i = 1; i > -2; i-=2) {
          for (int j = -1; j < 2; j+=2) {
            Dtype cx = (kernel_w - 1.0) / 2.0;
            Dtype cy = (kernel_h - 1.0) / 2.0;
            Dtype x = box[2*c+0]*cx;
            Dtype y = box[2*c+1]*cy;
            Dtype x_new = (T[0] + 1.0) * x + T[1] * y + T[2];
            Dtype y_new = T[3] * x + (T[4] + 1.0) * y + T[5];
            Dtype z_new = T[6] * x + T[7] * y + 1.0;
            z_new = 1.0;
            out_data[out->offset(n, 2 * c + 1, h, w)] = (y_new / z_new - (box[2*c+1] + 1) /2.0);
            out_data[out->offset(n, 2 * c + 0, h, w)] = (x_new / z_new - (box[2*c+0] + 1) /2.0);
            c += 1;
          }
        }
      }
    }
  }
}

template void transform_anchor(const Blob<float>* in,
    Blob<float>* out);
template void transform_anchor(const Blob<double>* in,
    Blob<double>* out);


template <typename TypeParam>
class TransformAnchorLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  TransformAnchorLayerTest()
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

  virtual ~TransformAnchorLayerTest() {
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
TYPED_TEST_CASE(TransformAnchorLayerTest, TestDtypesGPU);

TYPED_TEST(TransformAnchorLayerTest, TestSimpleTransformAnchor) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  shared_ptr<Layer<Dtype> > layer(
      new TransformAnchorLayer<Dtype>(layer_param));
  layer->Reshape(this->blob_bottom_vec_, this->blob_top_vec_);
  layer->Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  // Check against reference convolution.
  const Dtype* top_data;
  const Dtype* ref_top_data;
  transform_anchor(this->blob_bottom_, this->MakeReferenceTop(this->blob_top_));
  top_data = this->blob_top_->cpu_data();
  ref_top_data = this->ref_blob_top_->cpu_data();
  for (int i = 0; i < this->blob_top_->count(); ++i) {
    EXPECT_NEAR(top_data[i], ref_top_data[i], 1e-4);
  }
}
//Gradient
TYPED_TEST(TransformAnchorLayerTest, TestGradient) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  TransformAnchorLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_,this->blob_top_vec_);
  GradientChecker<Dtype> checker(1e-2, 1e-3);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_, this->blob_top_vec_);
}

}  // namespace caffe
