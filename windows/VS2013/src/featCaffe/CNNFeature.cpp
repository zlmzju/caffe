//#include <cuda_runtime.h>
#include <cstring>
#include <cstdlib>
#include <vector>
#include <string>
#include <iostream>
#include <stdio.h>
#include "caffe/caffe.hpp"
#include "caffe/util/io.hpp"
#include "caffe/blob.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <opencv2/imgproc/imgproc.hpp>
#include "boost/smart_ptr/shared_ptr.hpp"

#include "CNNFeature.h"

using namespace caffe;
using namespace std;
using namespace cv;
using namespace boost;


Net<float>  *net;
CNNFeature::CNNFeature(string protofile, string caffemodel, int useGPU_ID) // init the caffe model with prototxt file
{
	// Set GPU
	if(useGPU_ID>=0)
	{
		Caffe::set_mode(Caffe::GPU);
		int device_id = useGPU_ID;
		Caffe::SetDevice(device_id);
	}
	else
	{
		Caffe::set_mode(Caffe::CPU);
	}

	// Set to TEST Phase
	Caffe::set_phase(Caffe::TEST);
    // Load net
	net=new Net<float>(protofile);
    // Load pre-trained net (binary proto)
    net->CopyTrainedLayersFrom(caffemodel);
}
CNNFeature::~CNNFeature()
{
}
int CNNFeature::ExtractFeature(Mat &image,float *feature, int maxSize,string layerName)	// given image file name, return a feature vector
{
	if(!image.data)
	{
		return 0;
	}
	if (Caffe::phase() != Caffe::TEST)
	{
		// Set to TEST Phase
		Caffe::set_phase(Caffe::TEST);
		cout << endl << "Caffe:phase():" << Caffe::phase() << endl;
	}
    // Set vector for image
    vector<cv::Mat> imageVector;
    imageVector.push_back(image);

    // Set vector for label
    vector<int> labelVector;
    labelVector.push_back(0);//push_back 0 for initialize purpose

    // Net initialization
    boost::shared_ptr<MemoryDataLayer<float> > memory_data_layer;
    memory_data_layer = boost::static_pointer_cast<MemoryDataLayer<float>>(net->layer_by_name("data"));

	Datum datum;
	cv::Mat cv_img;
	cv::resize(image, cv_img, cv::Size(224, 224));
	CVMatToDatum(cv_img,&datum);

    vector<Datum> datums;
    for (int i = 0; i < 1; i++)
        datums.push_back(datum);
	memory_data_layer->AddDatumVector(datums);
	//Net forward
	float loss = 0.0;
    const vector<Blob<float>*>& results = net->ForwardPrefilled(&loss);
	//Feature
	const boost::shared_ptr<Blob<float> > feature_blob = net->blob_by_name(layerName);	//1*
	int dim_features = max(feature_blob->count(),maxSize);
	const float* blob_data=feature_blob->cpu_data() + feature_blob->offset(0);	//data of batch 0
	
	int memSize=sizeof(float)*dim_features;
	memset(feature,0,memSize);
	memcpy(feature,blob_data,memSize);
	//for (int d = 0; d < dim_features; ++d) 
	//{
	//	feature[d]=blob_data[d];
	//}
	return dim_features;	//feature dimension
}