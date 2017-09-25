#include <vector>
#include <iostream>

#include "caffe/layers/transform_offset_layer.hpp"
using namespace std;

namespace caffe {

/*input data_matrix, output data_offset
* input matrix T (size = 8) to projective matrix:
* T[0]+1, T[1]  , T[2]
* T[3]  , T[4]+1, T[5]
* T[6]  , T[7]  , 1
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

    const int w_old = c_off % kernel_w;
    const int h_old = (c_off / kernel_w) % kernel_h;
    //transform matrix multiplication: (3, 3) * (w_old, h_old, 1) = (h_new, w_new, z_new)
    Dtype h_new = 0;
    Dtype w_new = 0;
    Dtype z_new = 0;
    Dtype T[8]; //h0, h1, ..., h7, where h8 = 1
    for(int i = 0; i < 8; ++i){
            int mat_pos = (i * height_off + h_off) * width_off + w_off;
            T[i] = data_matrix[mat_pos];
    }

    h_new = (T[0] + 1.0) * w_old +         T[1] * h_old + T[2];
    w_new =         T[3] * w_old + (T[4] + 1.0) * h_old + T[5];
    z_new =         T[6] * w_old +         T[7] * h_old + 1;
    //assign new h and w to data_offset
    data_offset[2 * index + 0] = h_new / z_new - h_old;
    data_offset[2 * index + 1] = w_new / z_new - w_old;
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

    const int w_old = c_off % kernel_w;
    const int h_old = (c_off / kernel_w) % kernel_h;
    //transform matrix multiplication: (3, 3) * (w_old, h_old, 1) = (h_new, w_new, z_new)
    Dtype h_new = 0;
    Dtype w_new = 0;
    Dtype z_new = 0;
    Dtype T[8]; //h0, h1, ..., h7, where h8 = 1
    int idx[8]; //index for diff_matrix
    for(int i = 0; i < 8; ++i){
            idx[i] = (i * height_off + h_off) * width_off + w_off;
            T[i] = data_matrix[idx[i]];
    }

    h_new = (T[0] + 1.0) * w_old +         T[1] * h_old + T[2];
    w_new =         T[3] * w_old + (T[4] + 1.0) * h_old + T[5];
    z_new =         T[6] * w_old +         T[7] * h_old + 1;
    //assign gradients from diff_offset to diff_matrix
    dh = diff_offset[2 * index + 0];
    dw = diff_offset[2 * index + 1];

    diff_matrix[idx[0]] = dh * h_old / z_new;
    diff_matrix[idx[1]] = dh * w_old / z_new;
    diff_matrix[idx[2]] = dh *     1 / z_new;

    diff_matrix[idx[3]] = dw * h_old / z_new;
    diff_matrix[idx[4]] = dw * w_old / z_new;
    diff_matrix[idx[5]] = dw *     1 / z_new;

    diff_matrix[idx[6]] = - (dh * h_old * h_new + dw * w_old * w_new) / (z_new * z_new);
    diff_matrix[idx[7]] = - (dh * w_old * h_new + dw * h_old * w_new) / (z_new * z_new);
  }
}
template <typename Dtype>
void TransformOffsetLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
    const Dtype* matrix = bottom[0]->gpu_data();
    Dtype* offset = top[0]->mutable_gpu_data();

    const int num_threads = top[0]->count(1);
    for(int i = 0; i < top[0]->shape(0); ++i){
        matrix_to_offset<Dtype><<<CAFFE_GET_BLOCKS(num_threads), CAFFE_CUDA_NUM_THREADS>>>(
                num_threads, matrix, kernel_size, kernel_size, 
                top[0]->shape(2), top[0]->shape(3), offset);
    }
}

template <typename Dtype>
void TransformOffsetLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
   const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
    const Dtype* matrix = bottom[0]->gpu_data();
    const Dtype* offset_diff = top[0]->gpu_diff();
    Dtype* matrix_diff = bottom[0]->mutable_gpu_diff();

    const int num_threads = top[0]->count(1);
    for(int i = 0; i < top[0]->shape(0); ++i){
        offset_to_matrix<Dtype><<<CAFFE_GET_BLOCKS(num_threads), CAFFE_CUDA_NUM_THREADS>>>(
                num_threads, offset_diff, matrix, kernel_size, kernel_size, 
                top[0]->shape(2), top[0]->shape(3), matrix_diff);
    }

}
//

INSTANTIATE_LAYER_GPU_FUNCS(TransformOffsetLayer);

}  // namespace caffe
