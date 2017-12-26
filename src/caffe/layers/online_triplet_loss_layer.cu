#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layers/online_triplet_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void OnlineTripletLossLayer<Dtype>::Forward_gpu(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  // The forward pass computes the pairwise distances.
  const Dtype* bottom_data = bottom[0]->cpu_data();
  const Dtype* label = bottom[1]->cpu_data();
  Dtype* loss_data = top[0]->mutable_cpu_data();
  const int num = bottom[0]->num();
  const int dim = bottom[0]->count() / num;
  const int channels = bottom[0]->channels();
  const bool all_triplets = this->layer_param_.triplet_loss_param().all_triplets();

  /**
  * Computing the pairwise Euclidean distance
  */
  Blob<Dtype> temp_diff(1, channels, 1, 1);
  for (int i = 0; i<num; ++i) {
    Dtype* dist_row_i = dist_.mutable_cpu_data() + dist_.offset(i);
    for (int j = 0; j<num; ++j) {
      caffe_sub(
        channels,
        bottom_data + i*channels,
        bottom_data + j*channels,
        temp_diff.mutable_cpu_data());
      dist_row_i[j] = caffe_cpu_dot(channels, temp_diff.cpu_data(), temp_diff.cpu_data());
    }
  }


  /**
  * Find boundary of different classes.
  * A batch is composed by small groups with items belongs to the same class.
  *  e.g. gruop size is 3, batch size is 15:
  *    1 1 1 2 2 2 3 3 3 1 1 1 4 4 4
  */
  vector<int> boundary;
  Dtype prev = Dtype(-1);
  for (int i = 0; i<num; ++i) {
    if (prev != label[i]) {
      boundary.push_back(i);
      prev = label[i];
    }
  }
  boundary.push_back(num);

  //calculate mean distances within each class
  vector<Dtype> mean_distances;
  vector<pair<Dtype, Dtype> > min_max_distances;
  // classes
  for (int c = 0; c<boundary.size() - 1; c++) {
    // query
    Dtype sum_dist = 0.0;
    Dtype min_dist = 2.0;
    Dtype max_dist = 0.0;
    int pos_count = 0;
    for (int i = boundary[c]; i<boundary[c + 1]; ++i) {
      const Dtype * dist_data = dist_.cpu_data() + dist_.offset(i);
      for (int j = boundary[c]; j<boundary[c + 1]; ++j) {
        if (i == j) continue;
        if (dist_data[j] < min_dist) min_dist = dist_data[j];
        if (dist_data[j] > max_dist) max_dist = dist_data[j];
        sum_dist += dist_data[j];
        pos_count++;
      }
    }
    mean_distances.push_back(sum_dist / static_cast<Dtype>(pos_count));
    min_max_distances.push_back(pair<Dtype, Dtype>(min_dist, max_dist));
  }
  /**
  * Sampling triplets and computing the loss
  */
  Dtype pair_loss = Dtype(0);
  Dtype rank_loss = Dtype(0);
  Dtype cur_rank_loss;
  Dtype pos_dist;
  Dtype neg_dist;
 
  //triplet loss
  triplets_.clear();
  int all_triplet_size = 0;
  int num_error = 0;
  // classes
  for (int c = 0; c<boundary.size() - 1; c++) {
    // query
    Dtype hard_pos_threshold = 2 * mean_distances[c] - min_max_distances[c].first;
    for (int i = boundary[c]; i<boundary[c + 1]; ++i) {
      const Dtype * dist_data = dist_.cpu_data() + dist_.offset(i);
      // positive
      for (int j = boundary[c]; j<boundary[c + 1]; ++j) {
        if (i == j) {
          continue;
        }
        pos_dist = dist_data[j];
        switch (this->layer_param_.triplet_loss_param().positive()){
        case TripletLossParameter_SampleMethod_ALL:
          break;
        case TripletLossParameter_SampleMethod_HARD:
          //sample the positives whose distance greater than the average
          if (pos_dist < mean_distances[c]) continue;
          break;
        case TripletLossParameter_SampleMethod_MODERATE:
          //sample the positives without the hardest ones
          if (pos_dist > hard_pos_threshold) continue;
          break;
        default:
          LOG(FATAL) << "Unknown positive sampling method.";
        }

        // negative groups
        for (int m = 0; m<boundary.size() - 1; m++) {
          if (label[boundary[m]] == label[i]) {
            continue;
          }
          // negative
          for (int k = boundary[m]; k<boundary[m + 1]; ++k) {
            all_triplet_size++;
            neg_dist = dist_data[k];
            cur_rank_loss = margin_ + pos_dist - neg_dist;
            num_error += (pos_dist >= neg_dist);
            if (cur_rank_loss < 0) continue;  //not violate the loss
            switch (this->layer_param_.triplet_loss_param().negative()){
            case TripletLossParameter_SampleMethod_ALL:
              break;
            case TripletLossParameter_SampleMethod_HARD:
              //sample the negatives whose distance smaller than the positive one
              if (neg_dist > pos_dist) continue;
              break;
            case TripletLossParameter_SampleMethod_MODERATE:
              //sample the positives without the hardest ones
              if (neg_dist <= pos_dist) continue;
              break;
            default:
              LOG(FATAL) << "Unknown negative sampling method.";
            }
            rank_loss += cur_rank_loss;
            triplets_.push_back(Triplet(i, j, k));
          } // end of negative
        } // end of negative groups
      } // end of positive
    } // end of query
  } // end of classes
  //original num_triplets_ is too large, but triplets_.size() would cause gradient non-smooth
  if (all_triplets)
    num_triplets_ = static_cast<Dtype>(all_triplet_size); //triplets_.size(); //
  else
    num_triplets_ = static_cast<Dtype>(triplets_.size());
  rank_loss = num_triplets_> 0 ? rank_loss / num_triplets_ : 0;

  // average loss among all triplets
  loss_data[0] = rank_loss * mu_;
  // average accuracy among all triplets
  if (top.size()>1)
    top[1]->mutable_cpu_data()[0] = Dtype(1) - (all_triplet_size > 0 ? Dtype(num_error) / all_triplet_size : 0);
}

template <typename Dtype>
void OnlineTripletLossLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
  const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[1]) {
    LOG(FATAL) << this->type()
      << " Layer cannot backpropagate to label inputs.";
  }
  if (propagate_down[0]) {
    const Dtype* bottom_data = bottom[0]->gpu_data();
    Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
    int count = bottom[0]->count();
    int num = bottom[0]->num();
    int dim = count / num;
    Dtype * agg_data = (Dtype *)aggregator_->mutable_cpu_data();
    caffe_memset(num * num * sizeof(Dtype), 0, agg_data);

    if (num_triplets_ > 0){
      Dtype scale = Dtype(2.0) / num_triplets_ * mu_;
      for (int i = 0; i<triplets_.size(); ++i) {
        int qry_id = triplets_[i].first_;
        int pos_id = triplets_[i].second_;
        int neg_id = triplets_[i].third_;

        agg_data[qry_id * num + neg_id] += scale;
        agg_data[qry_id * num + pos_id] -= scale;

        agg_data[pos_id * num + pos_id] += scale;
        agg_data[pos_id * num + qry_id] -= scale;

        agg_data[neg_id * num + qry_id] += scale;
        agg_data[neg_id * num + neg_id] -= scale;
      }
    }
    const Dtype * agg_gpu_data = (Dtype *)aggregator_->gpu_data();
    const Dtype loss_weight = top[0]->cpu_diff()[0];
    caffe_gpu_gemm(CblasNoTrans, CblasNoTrans, num, dim, num,
      loss_weight, agg_gpu_data, bottom_data, Dtype(0), bottom_diff);
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(OnlineTripletLossLayer);

}  // namespace caffe
