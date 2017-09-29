#include <vector>
#include <iostream>

#include "caffe/util/gpu_util.cuh"
#include "caffe/layers/transform_rs_layer.hpp"
using namespace std;

namespace caffe {

/*input data_matrix, output data_offset
* input matrix T (size = 3) to projective matrix:
* theta = T[0], Sx = (T[1] + 1.0), Sy=(T[2] + 1.0)
*/

template <typename Dtype>
__global__ void matrix_to_offset(const int n, const Dtype* data_matrix,
  const int kernel_h, const int kernel_w,
  const int height_off, const int width_off,
  Dtype* data_offset) {
  CUDA_KERNEL_LOOP(index, n) { //n is the size of (kernel_h*kernel_w, h, w)
    // index of offset and tranform matrix
    const int w_off = index % width_off;
    const int h_off = (index / width_off) % height_off;
    const int c_off = index / width_off / height_off;

    const Dtype w_old = Dtype(c_off % kernel_w) - (kernel_w -1.0) / 2.0;
    const Dtype h_old = Dtype((c_off / kernel_w) % kernel_h) - (kernel_h -1.0) / 2.0;
    //transform matrix multiplication: (3, 3) * (w_old, h_old, 1) = (h_new, w_new, z_new)
    Dtype T[3]; // T[0] = theta, T[1] = Sx, T[2] = Sy
    int idx[3]; //index for diff_matrix
    for(int i = 0; i < 3; ++i){
        idx[i] = (i * height_off + h_off) * width_off + w_off;
        T[i] = data_matrix[idx[i]];
    }

    Dtype s_h_old = (T[1] + 1.0) * h_old;
    Dtype s_w_old = (T[2] + 1.0) * w_old;

    Dtype h_new = cos(T[0]) * s_h_old - sin(T[0]) * s_w_old;
    Dtype w_new = sin(T[0]) * s_h_old + cos(T[0]) * s_w_old;
    //assign new h and w to data_offset
    int offset_index_h = ((2 * c_off + 0) * height_off + h_off) * width_off + w_off;
    int offset_index_w = ((2 * c_off + 1) * height_off + h_off) * width_off + w_off;
    data_offset[offset_index_h] = h_new - h_old;
    data_offset[offset_index_w] = w_new - w_old;
  }
}

template <typename Dtype>
__global__ void offset_to_matrix(const int n, 
  const Dtype* diff_offset, const Dtype* data_matrix,
  const int kernel_h, const int kernel_w,
  const int height_off, const int width_off,
  Dtype* diff_matrix) {
  CUDA_KERNEL_LOOP(index, n) { //n is the size of (kernel_h*kernel_w, h, w)
    // index of offset and tranform matrix
    const int w_off = index % width_off;
    const int h_off = (index / width_off) % height_off;
    const int c_off = index / width_off / height_off;

    const Dtype w_old = Dtype(c_off % kernel_w) - (kernel_w -1.0) / 2.0;
    const Dtype h_old = Dtype((c_off / kernel_w) % kernel_h) - (kernel_h -1.0) / 2.0;
    //transform matrix multiplication: (3, 3) * (w_old, h_old, 1) = (h_new, w_new, z_new)
    Dtype T[3]; //T[0] = theta, T[1] = Sx, T[2] = Sy
    int idx[3]; //index for diff_matrix
    for(int i = 0; i < 3; ++i){
        idx[i] = (i * height_off + h_off) * width_off + w_off;
        T[i] = data_matrix[idx[i]];
    }

    Dtype s_h_old = (T[1] + 1.0) * h_old;
    Dtype s_w_old = (T[2] + 1.0) * w_old;

    Dtype h_new = cos(T[0]) * s_h_old - sin(T[0]) * s_w_old;
    Dtype w_new = sin(T[0]) * s_h_old + cos(T[0]) * s_w_old;
    //assign new h and w to data_offset
    int offset_index_h = ((2 * c_off + 0) * height_off + h_off) * width_off + w_off;
    int offset_index_w = ((2 * c_off + 1) * height_off + h_off) * width_off + w_off;
    Dtype dh = diff_offset[offset_index_h];
    Dtype dw = diff_offset[offset_index_w];

    //diff matrix values
    Dtype D[3];
    D[0] = dh * ((-sin(T[0])) * s_h_old - cos(T[0]) * s_w_old);
    D[0]+= dw * (cos(T[0]) * s_h_old + (-sin(T[0])) * s_w_old);
    D[1] = dh * cos(T[0]) * h_old + dw * sin(T[0]) * h_old;
    D[2] = dh * (-sin(T[0]) * w_old) + dw * cos(T[0]) * w_old;

    //atomic add
    for(int i = 0; i < 3; ++i){
        caffe_gpu_atomic_add(D[i], diff_matrix + idx[i]);
    }
  }
}
template <typename Dtype>
void TransformRSLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
    const Dtype* matrix = bottom[0]->gpu_data();
    Dtype* offset = top[0]->mutable_gpu_data();

    const int num_threads = top[0]->count(1) / 2;
    for(int i = 0; i < top[0]->shape(0); ++i){
        matrix_to_offset<Dtype><<<CAFFE_GET_BLOCKS(num_threads), CAFFE_CUDA_NUM_THREADS>>>(
                num_threads, matrix + i * bottom[0]->count(1), this->kernel_size, this->kernel_size, 
                top[0]->shape(2), top[0]->shape(3), offset + i * top[0]->count(1));
    }
}

template <typename Dtype>
void TransformRSLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
   const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
    const Dtype* matrix = bottom[0]->gpu_data();
    const Dtype* offset_diff = top[0]->gpu_diff();
    Dtype* matrix_diff = bottom[0]->mutable_gpu_diff();
    caffe_gpu_set(bottom[0]->count(), Dtype(0), matrix_diff);

    const int num_threads = top[0]->count(1) / 2;
    for(int i = 0; i < top[0]->shape(0); ++i){
        offset_to_matrix<Dtype><<<CAFFE_GET_BLOCKS(num_threads), CAFFE_CUDA_NUM_THREADS>>>(
                num_threads, offset_diff + i * top[0]->count(1), matrix + i * bottom[0]->count(1),
                this->kernel_size, this->kernel_size, 
                top[0]->shape(2), top[0]->shape(3), 
                matrix_diff + i * bottom[0]->count(1));
    }

}
//

INSTANTIATE_LAYER_GPU_FUNCS(TransformRSLayer);

}  // namespace caffe
