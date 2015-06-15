/*======================================================================================
//
// Author : Liming Zhao
// E-mail : zhaoliming@zju.edu.cn
// Website: http://www.zhaoliming.net
//
// Last modified: 2015-06-13 15:53
//
// Filename: guidedfilter_layer.cpp
//
// Description: 
// Guided Image Filter Layer
// input:
//			1, gray image
//			2, coarse map
// output:
//			finegrained map
=====================================================================================*/
#include <algorithm>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void GuidedFilterLayer<Dtype>::MeanFilter(const Dtype* src, Dtype* dst)
{
    Dtype* input_data=pool_input_.mutable_cpu_data();
    caffe_copy(pool_input_.count(),src,input_data);
    pool_layer_->Forward(pool_input_vec_,pool_output_vec_);

    const Dtype* output_data=pool_output_.cpu_data();
    caffe_copy(pool_output_.count(),output_data,dst);
}


template <typename Dtype>
void GuidedFilterLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top)
{
    //parameters of this
    window_size_=this->layer_param_.guided_filter_param().window_size();       
    epsilon_=this->layer_param_.guided_filter_param().epsilon();    //defined in ../proto/caffe.proto
    
    //parameters of pool_layer used to compute mean
    LayerParameter pool_param;
    pool_param.mutable_pooling_param()->set_pool(
		    PoolingParameter_PoolMethod_AVE);
    pool_param.mutable_pooling_param()->set_pad(window_size_/2);
    pool_param.mutable_pooling_param()->set_kernel_size(1+2*(window_size_-1));
    pool_param.mutable_pooling_param()->set_stride(1);	//even default is 1
    pool_layer_.reset(new PoolingLayer<Dtype>(pool_param));

    //set up pool_layer
    pool_input_.Reshape(1,1,bottom[1]->height(),bottom[1]->width());
    pool_input_vec_.clear();
    pool_input_vec_.push_back(&pool_input_);
    pool_output_vec_.clear();
    pool_output_vec_.push_back(&pool_output_);
    pool_layer_->SetUp(pool_input_vec_,pool_output_vec_);
}

template <typename Dtype>
void GuidedFilterLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) 
{
  CHECK(bottom[1]->channels()==1)<<
	  "GuidedFilterLayer takes bottom[1] with only 1 channel map.";
  top[0]->ReshapeLike(*bottom[0]);
  pool_input_.Reshape(1,1,bottom[1]->height(),bottom[1]->width());
  pool_layer_->Reshape(pool_input_vec_,pool_output_vec_);
}

template <typename Dtype>
void GuidedFilterLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) 
{
  const Dtype* I_all=bottom[0]->cpu_data();
  const Dtype* p=bottom[1]->cpu_data();
  Dtype* top_data_all=top[0]->mutable_cpu_data();
  int count=bottom[1]->count();	//make sure count=height*width
  Dtype* mean_a_=new Dtype[count];
  Dtype* mean_b_=new Dtype[count];
  //the main loop
  for(int n=0;n<bottom[0]->num();n++)
  {
      for(int c=0;c<bottom[0]->channels();c++)	//current version channels()==1
      {
        const Dtype* I=I_all+bottom[0]->offset(n,c);
        Dtype* top_data=top_data_all+top[0]->offset(n,c);
        Dtype* mean_I=new Dtype[count];
        MeanFilter(I, mean_I); //mean value for each pixel in one window
        //caffe_copy(count,mean_I,top_data);
        
        Dtype* mean_p=new Dtype[count];
        MeanFilter(p, mean_p);

        Dtype* I_mul_p=new Dtype[count];
        caffe_mul(count,I,p,I_mul_p);

        Dtype* mean_Ip=new Dtype[count];
        MeanFilter(I_mul_p, mean_Ip);

        Dtype* meanI_mul_meanp=new Dtype[count];
        caffe_mul(count,mean_I,mean_p,meanI_mul_meanp);

        Dtype* cov_Ip=mean_Ip;
        caffe_sub(count,mean_Ip,meanI_mul_meanp,cov_Ip);
        
        Dtype*  I_mul_I=I_mul_p;
        caffe_mul(count,I,I,I_mul_I);

        Dtype* mean_II=new Dtype[count];
        MeanFilter(I_mul_I, mean_II);

        Dtype* meanI_mul_meanI=meanI_mul_meanp;
        caffe_mul(count,mean_I,mean_I,meanI_mul_meanI);

        Dtype* var_I=mean_II;
        caffe_sub(count,mean_II,meanI_mul_meanI,var_I);

        //a=cov_Ip./(var_I+eps)
        //b=mean_p-a.*mean_I
        Dtype* a=new Dtype[count];
        
        Dtype* var_I_eps=var_I;
        caffe_add_scalar(count,(Dtype)epsilon_,var_I_eps);

        caffe_div(count,cov_Ip,var_I_eps,a);

        Dtype* a_mul_meanI=I_mul_I;	//just reuse old memory
        caffe_mul(count,a,mean_I,a_mul_meanI);

        Dtype* b=new Dtype[count];
        caffe_sub(count,mean_p,a_mul_meanI,b);

        //TODO: add private parameters mean_a_ and mean_b_
        MeanFilter(a, mean_a_);
        MeanFilter(b, mean_b_);
        
        //output top
        caffe_mul(count,mean_a_,I,top_data);
        caffe_add(count,top_data,mean_b_,top_data);
      }
  }
}

template <typename Dtype>
void GuidedFilterLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) 
{
  NOT_IMPLEMENTED;
}

//gpu version not implemented
template <typename Dtype>
void GuidedFilterLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) 
{
  //NOT_IMPLEMENTED;
  Forward_cpu(bottom,top);
}


template <typename Dtype>
void GuidedFilterLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) 
{
  //NOT_IMPLEMENTED;
  Backward_cpu(top,propagate_down,bottom);
}


#ifdef CPU_ONLY
STUB_GPU(GuidedFilterLayer);
#endif

INSTANTIATE_CLASS(GuidedFilterLayer);
REGISTER_LAYER_CLASS(GuidedFilter);

}  // namespace caffe
