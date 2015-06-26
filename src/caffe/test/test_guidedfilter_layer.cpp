#include <cstring>
#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/vision_layers.hpp"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

namespace caffe {

template <typename TypeParam>
class GuidedFilterLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  GuidedFilterLayerTest()
      : blob_bottom_I_(new Blob<Dtype>(2,2,5,5)),
	blob_bottom_p_(new Blob<Dtype>(2,1,5,5)),
        blob_top_q_(new Blob<Dtype>()) 
  {
    blob_bottom_vec_.push_back(blob_bottom_I_);
    blob_bottom_vec_.push_back(blob_bottom_p_);
    blob_top_vec_.push_back(blob_top_q_);
  }
  virtual ~GuidedFilterLayerTest() {
    delete blob_bottom_I_;
    delete blob_bottom_p_;
    delete blob_top_q_;
  }
  Blob<Dtype>* blob_bottom_I_;
  Blob<Dtype>* const blob_bottom_p_;
  Blob<Dtype>* const blob_top_q_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;

  void TestForward()
  {
    //init the layer
    LayerParameter layer_param;
    GuidedFilterParameter* guided_filter_param = layer_param.mutable_guided_filter_param();
    guided_filter_param->set_window_size(2);
    GuidedFilterLayer<Dtype> layer(layer_param);

    //init the bottom data
    //I: 2x1 channels of 5x5:
    //	[0.780016225238194   0.749580424362298	0.740414372089090   0.741393586774514	0.751388236837560]
    //
    //	[0.737811008594189   0.681265649729138	0.659440609971576   0.616152738174030	0.587378842763567]
    //
    //	[0.563849430998861   0.537335198750417	0.532537855905660   0.557046482355790	0.579252979102189]
    //
    //	[0.572817689949016   0.577676006720180	0.596196484818137   0.599882503466579	0.595320710600831]
    //
    //	[0.608940405385914   0.592571440096101	0.608300181147616   0.600457043892714	0.585665050984911]
    //
    //p: 2x1 channels of 5x5:
    //	[0.564705882352941   0.615686274509804	0.619607843137255   0.611764705882353	0.592156862745098]
    //
    //	[0.721568627450980   0.690196078431373	0.690196078431373   0.752941176470588	0.729411764705882]
    //
    //	[0.788235294117647   0.780392156862745	0.827450980392157   0.780392156862745	0.768627450980392]
    //
    //	[0.827450980392157   0.843137254901961	0.843137254901961   0.847058823529412	0.874509803921569]
    //
    //	[0.854901960784314   0.921568627450980	0.866666666666667   0.866666666666667	0.854901960784314]
    //
 
    const int num=2;
    const int channels=2;
    const int height=5;
    const int width=5;
    Dtype *bottom_I_data=NULL;
    Dtype *bottom_p_data=NULL;
    //blob_bottom_I_->Reshape(num,channels,height,width);
    //blob_bottom_p_->Reshape(num,1,height,width);

    if(Caffe::mode()==Caffe::CPU)
    {
        bottom_I_data=blob_bottom_I_->mutable_cpu_data();
        bottom_p_data=blob_bottom_p_->mutable_cpu_data();
    }
    else if(Caffe::mode()==Caffe::GPU)
    {   
        bottom_I_data=blob_bottom_I_->mutable_gpu_data();
        bottom_p_data=blob_bottom_p_->mutable_gpu_data();
    }
    else
    {
        //something error
        EXPECT_EQ(1,2);
    }
    Dtype *bottom_I_data_temp=new Dtype[blob_bottom_I_->count()];
    Dtype *bottom_p_data_temp=new Dtype[blob_bottom_p_->count()];

    for(int i=0;i<num*channels*height*width;i+=height*width)
    {
	//input gray image I
	bottom_I_data_temp[i+0]=0.780016225238193;
	bottom_I_data_temp[i+1]=0.749580424362298;
	bottom_I_data_temp[i+2]=0.740414372089090;
	bottom_I_data_temp[i+3]=0.741393586774514;
	bottom_I_data_temp[i+4]=0.751388236837560;
	bottom_I_data_temp[i+5]=0.737811008594189;
	bottom_I_data_temp[i+6]=0.681265649729138;
	bottom_I_data_temp[i+7]=0.659440609971576;
	bottom_I_data_temp[i+8]=0.616152738174030;
	bottom_I_data_temp[i+9]=0.587378842763567;
	bottom_I_data_temp[i+10]=0.563849430998861;
	bottom_I_data_temp[i+11]=0.537335198750416;
	bottom_I_data_temp[i+12]=0.532537855905660;
	bottom_I_data_temp[i+13]=0.557046482355789;
	bottom_I_data_temp[i+14]=0.579252979102189;
	bottom_I_data_temp[i+15]=0.572817689949016;
	bottom_I_data_temp[i+16]=0.577676006720180;
	bottom_I_data_temp[i+17]=0.596196484818137;
	bottom_I_data_temp[i+18]=0.599882503466579;
	bottom_I_data_temp[i+19]=0.595320710600831;
	bottom_I_data_temp[i+20]=0.608940405385914;
	bottom_I_data_temp[i+21]=0.592571440096101;
	bottom_I_data_temp[i+22]=0.608300181147616;
	bottom_I_data_temp[i+23]=0.600457043892714;
	bottom_I_data_temp[i+24]=0.585665050984911;
    }
    for(int i=0;i<num*1*height*width;i+=height*width)
    {
        //input coarse map p
	bottom_p_data_temp[i+0]=0.564705882352941;
	bottom_p_data_temp[i+1]=0.615686274509804;
	bottom_p_data_temp[i+2]=0.619607843137255;
	bottom_p_data_temp[i+3]=0.611764705882353;
	bottom_p_data_temp[i+4]=0.592156862745098;
	bottom_p_data_temp[i+5]=0.721568627450980;
	bottom_p_data_temp[i+6]=0.690196078431373;
	bottom_p_data_temp[i+7]=0.690196078431373;
	bottom_p_data_temp[i+8]=0.752941176470588;
	bottom_p_data_temp[i+9]=0.729411764705882;
	bottom_p_data_temp[i+10]=0.788235294117647;
	bottom_p_data_temp[i+11]=0.780392156862745;
	bottom_p_data_temp[i+12]=0.827450980392157;
	bottom_p_data_temp[i+13]=0.780392156862745;
	bottom_p_data_temp[i+14]=0.768627450980392;
	bottom_p_data_temp[i+15]=0.827450980392157;
	bottom_p_data_temp[i+16]=0.843137254901961;
	bottom_p_data_temp[i+17]=0.843137254901961;
	bottom_p_data_temp[i+18]=0.847058823529412;
	bottom_p_data_temp[i+19]=0.874509803921569;
	bottom_p_data_temp[i+20]=0.854901960784314;
	bottom_p_data_temp[i+21]=0.921568627450980;
	bottom_p_data_temp[i+22]=0.866666666666667;
	bottom_p_data_temp[i+23]=0.866666666666667;
	bottom_p_data_temp[i+24]=0.854901960784314;
    }
    caffe_copy(blob_bottom_I_->count(),bottom_I_data_temp,bottom_I_data);
    caffe_copy(blob_bottom_p_->count(),bottom_p_data_temp,bottom_p_data);
    //setup layer and forward
    layer.SetUp(blob_bottom_vec_,blob_top_vec_);
    EXPECT_EQ(blob_top_q_->num(),num);
    EXPECT_EQ(blob_top_q_->channels(),channels);
    EXPECT_EQ(blob_top_q_->height(),height);
    EXPECT_EQ(blob_top_q_->width(),width);

    layer.Forward(blob_bottom_vec_,blob_top_vec_);

    //expected output is q (use matlab code)
    //q: 2x1 channels of 5x5:
    //	[0.305451764936534	0.445733832171568   0.437143672489376	0.461166976004217   0.316816551726667]
    //
    //	[0.474597728734006	0.685150618246553   0.682412051907470	0.696731160483050   0.457467394868920]
    //
    //	[0.501004634015045	0.762760047468481   0.812850720701027	0.776202999019964   0.514125286875935]
    //
    //	[0.539212757413632	0.817764745227509   0.834357568317334	0.837447876355424   0.555031486259445]
    //
    //	[0.392313547968571	0.573363834433404   0.582377960487622	0.574635206634875   0.373106238623831]
    const Dtype *top_data=NULL;
    Dtype *top_data_temp=NULL;
    if(Caffe::mode()==Caffe::CPU)
    {
        top_data=blob_top_q_->cpu_data();
    }
    else if(Caffe::mode()==Caffe::GPU)
    {   
        top_data_temp=new Dtype[blob_top_q_->count()];
        caffe_copy(blob_top_q_->count(),blob_top_q_->gpu_data(),top_data_temp);
        top_data=top_data_temp;
    }
    else
    {
        //something error
        EXPECT_EQ(1,2);
    }

    //EXPECT_NEAR(top_data[0], 0.305451764936534, 1e-5);

    for(int i=0;i<num*channels*height*width;i+=height*width)
    {
	EXPECT_NEAR(top_data[i + 0], 0.305451764936534, 1e-5);
	EXPECT_NEAR(top_data[i + 1], 0.445733832171568, 1e-5);
	EXPECT_NEAR(top_data[i + 2], 0.437143672489376, 1e-5);
	EXPECT_NEAR(top_data[i + 3], 0.461166976004217, 1e-5);
	EXPECT_NEAR(top_data[i + 4], 0.316816551726667, 1e-5);
	EXPECT_NEAR(top_data[i + 5], 0.474597728734006, 1e-5);
	EXPECT_NEAR(top_data[i + 6], 0.685150618246553, 1e-5);
	EXPECT_NEAR(top_data[i + 7], 0.682412051907470, 1e-5);
	EXPECT_NEAR(top_data[i + 8], 0.696731160483050, 1e-5);
	EXPECT_NEAR(top_data[i + 9], 0.457467394868920, 1e-5);
	EXPECT_NEAR(top_data[i + 10], 0.501004634015045, 1e-5);
	EXPECT_NEAR(top_data[i + 11], 0.762760047468481, 1e-5);
	EXPECT_NEAR(top_data[i + 12], 0.812850720701027, 1e-5);
	EXPECT_NEAR(top_data[i + 13], 0.776202999019964, 1e-5);
	EXPECT_NEAR(top_data[i + 14], 0.514125286875935, 1e-5);
	EXPECT_NEAR(top_data[i + 15], 0.539212757413632, 1e-5);
	EXPECT_NEAR(top_data[i + 16], 0.817764745227509, 1e-5);
	EXPECT_NEAR(top_data[i + 17], 0.834357568317334, 1e-5);
	EXPECT_NEAR(top_data[i + 18], 0.837447876355424, 1e-5);
	EXPECT_NEAR(top_data[i + 19], 0.555031486259445, 1e-5);
	EXPECT_NEAR(top_data[i + 20], 0.392313547968571, 1e-5);
	EXPECT_NEAR(top_data[i + 21], 0.573363834433404, 1e-5);
	EXPECT_NEAR(top_data[i + 22], 0.582377960487621, 1e-5);
	EXPECT_NEAR(top_data[i + 23], 0.574635206634875, 1e-5);
	EXPECT_NEAR(top_data[i + 24], 0.373106238623831, 1e-5);
    }
  }
};

//../../include/caffe/test/test_caffe_main.hpp
typedef ::testing::Types<FloatCPU,FloatGPU> TestFloat;

//TYPED_TEST_CASE(GuidedFilterLayerTest, TestDtypesAndDevices);
TYPED_TEST_CASE(GuidedFilterLayerTest,TestFloat);

TYPED_TEST(GuidedFilterLayerTest, TestForward)
{
    this->TestForward();
}

/*
TYPED_TEST(GuidedFilterLayerTest, TestSetup) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  GuidedFilterParameter* guided_filter_param = layer_param.mutable_guided_filter_param();
  guided_filter_param->set_window_size(3);
  GuidedFilterLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_->num(), this->blob_bottom_->num());
  EXPECT_EQ(this->blob_top_->channels(), this->blob_bottom_->channels());
  EXPECT_EQ(this->blob_top_->height(), 3);
  EXPECT_EQ(this->blob_top_->width(), 2);
}

TYPED_TEST(GuidedFilterLayerTest, TestGradientAve) {
  typedef typename TypeParam::Dtype Dtype;
  for (int kernel_h = 3; kernel_h <= 4; kernel_h++) {
    for (int kernel_w = 3; kernel_w <= 4; kernel_w++) {
      LayerParameter layer_param;
      GuidedFilterParameter* guided_filter_param = layer_param.mutable_guided_filter_param();
      guided_filter_param->set_kernel_h(kernel_h);
      guided_filter_param->set_kernel_w(kernel_w);
      guided_filter_param->set_stride(2);
      guided_filter_param->set_pool(GuidedFilterParameter_PoolMethod_AVE);
      GuidedFilterLayer<Dtype> layer(layer_param);
      GradientChecker<Dtype> checker(1e-2, 1e-2);
      checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
          this->blob_top_vec_);
    }
  }
}

TYPED_TEST(GuidedFilterLayerTest, TestGradientAvePadded) {
  typedef typename TypeParam::Dtype Dtype;
  for (int kernel_h = 3; kernel_h <= 4; kernel_h++) {
    for (int kernel_w = 3; kernel_w <= 4; kernel_w++) {
      LayerParameter layer_param;
      GuidedFilterParameter* guided_filter_param = layer_param.mutable_guided_filter_param();
      guided_filter_param->set_kernel_h(kernel_h);
      guided_filter_param->set_kernel_w(kernel_w);
      guided_filter_param->set_stride(2);
      guided_filter_param->set_pad(2);
      guided_filter_param->set_pool(GuidedFilterParameter_PoolMethod_AVE);
      GuidedFilterLayer<Dtype> layer(layer_param);
      GradientChecker<Dtype> checker(1e-2, 1e-2);
      checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
          this->blob_top_vec_);
    }
  }
}
*/
}  // namespace caffe
