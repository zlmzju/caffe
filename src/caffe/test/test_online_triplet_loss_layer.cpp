#include <cmath>
#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layers/online_triplet_loss_layer.hpp"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

namespace caffe {

template <typename TypeParam>
class OnlineTripletLossLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

protected:
  OnlineTripletLossLayerTest()
    : blob_bottom_data_(new Blob<Dtype>(10, 5, 1, 1)),
    blob_bottom_label_(new Blob<Dtype>(10, 1, 1, 1)),
    blob_top_loss_(new Blob<Dtype>()) {
    // fill the values
    FillerParameter filler_param;
    filler_param.set_min(-2.0);
    filler_param.set_max(2.0);  
    UniformFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_data_);
    blob_bottom_vec_.push_back(blob_bottom_data_);
    //fill label with 5 ids, gruop_size_= 2 (10/5)
    for (int i = 0; i < blob_bottom_label_->count(); ++i) {
      blob_bottom_label_->mutable_cpu_data()[i] = i / group_size_;  // {0,0,1,1,2,2,...,4,4}
    }
    blob_bottom_vec_.push_back(blob_bottom_label_);
    blob_top_vec_.push_back(blob_top_loss_);
  }

  virtual ~OnlineTripletLossLayerTest() {
    delete blob_bottom_data_;
    delete blob_bottom_label_;
    delete blob_top_loss_;
  }

  void TestForward() {
    LayerParameter layer_param;
    OnlineTripletLossLayer<Dtype> layer(layer_param);
    // manually compute to compare
    const Dtype margin = layer_param.triplet_loss_param().margin();
    const int num = this->blob_bottom_data_->num();
    const int channels = this->blob_bottom_data_->channels();
    const int class_num = blob_bottom_label_->count() / group_size_;
    //calculate pairwise Euclidean distances
    Blob<Dtype> distances(num, num, 1, 1);
    Blob<Dtype> temp_diff(1, channels, 1, 1);
    const Dtype* bottom_data = blob_bottom_data_->cpu_data();
    for (int i = 0; i<num; ++i) {
      Dtype* dist_row_i = distances.mutable_cpu_data() + distances.offset(i);
      for (int j = 0; j<num; ++j) {
        caffe_sub(
          channels,
          bottom_data + i*channels,
          bottom_data + j*channels,
          temp_diff.mutable_cpu_data());
        dist_row_i[j] = caffe_cpu_dot(channels, temp_diff.cpu_data(), temp_diff.cpu_data());
      }
    }
    //loss
    Dtype loss(0);
    int num_triplets = 0;
    int all_triplets_size = 0;
    // classes
    for (int c = 0; c < class_num; c++) {
      // query
      for (int i = 0; i<group_size_; ++i) {
        const int query_idx = c*group_size_ + i;
        const int query_label = blob_bottom_label_->data_at(query_idx, 0, 0, 0);
        const Dtype * dist_data = distances.cpu_data() + distances.offset(query_idx);
        // positive
        for (int j = 0; j<group_size_; ++j) {
          if (i == j) {
            continue;
          }
          const int pos_idx = c*group_size_ + j;
          Dtype pos_dist = dist_data[pos_idx];

          // negative groups
          for (int m = 0; m < class_num; m++) {
            for (int k = 0; k<group_size_; ++k) {
              const int neg_idx = m*group_size_ + k;
              if (query_label == blob_bottom_label_->data_at(neg_idx, 0, 0, 0)) {
                continue;
              }
              // negative
              all_triplets_size++;
              Dtype neg_dist = dist_data[neg_idx];
              Dtype cur_rank_loss = margin + pos_dist - neg_dist;
              if (cur_rank_loss > 0) {
                ++num_triplets;
                loss += cur_rank_loss;
              }
            } // end of negative
          } // end of negative groups
        } // end of positive
      } // end of query
    } // end of classes
    loss = num_triplets > 0 ? loss / num_triplets : 0;


    layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
    layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
    EXPECT_NEAR(this->blob_top_loss_->cpu_data()[0], loss, 1e-6);
  }

  int group_size_ = 2;
  Blob<Dtype>* const blob_bottom_data_;
  Blob<Dtype>* const blob_bottom_label_;
  Blob<Dtype>* const blob_top_loss_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(OnlineTripletLossLayerTest, TestDtypesAndDevices);

TYPED_TEST(OnlineTripletLossLayerTest, TestForward) {
  this->TestForward();
}

TYPED_TEST(OnlineTripletLossLayerTest, TestGradient) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  const Dtype kLossWeight = 3.7;
  layer_param.add_loss_weight(kLossWeight);
  TripletLossParameter* triplet_loss_param = layer_param.mutable_triplet_loss_param();
  triplet_loss_param->set_all_triplets(true);
  OnlineTripletLossLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  GradientChecker<Dtype> checker(1e-2, 1e-2, 1701); 
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
    this->blob_top_vec_,0);
}

}  // namespace caffe
