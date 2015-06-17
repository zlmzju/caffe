/*
#include <cuda_runtime.h>
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
#include "boost/smart_ptr/shared_ptr.hpp"

using namespace caffe;
using namespace std;
using namespace cv;
using namespace boost;

int main(int argc, char** argv)
{
    // Set GPU
    Caffe::set_mode(Caffe::GPU);
    int device_id = 0;
    Caffe::SetDevice(device_id);
    LOG(INFO) << "Using GPU";

    // Set to TEST Phase
    Caffe::set_phase(Caffe::TEST);

    // Load net
    Net<float> net("./examples/fcn/fcn-32s-pascal-memory.prototxt");

    // Load pre-trained net (binary proto)
    net.CopyTrainedLayersFrom("./examples/fcn/fcn-32s-pascal.caffemodel");

    // Load image
    //string imgName = "../examples/images/fish-bike.jpg";
    string imgName = "./examples/images/cat2.jpg";
    Mat image = imread(imgName);
    imshow("image", image);

    // Set vector for image
    vector<cv::Mat> imageVector;
    imageVector.push_back(image);

    // Set vector for label
    vector<int> labelVector;
    labelVector.push_back(0);//push_back 0 for initialize purpose

    // Net initialization
    float loss = 0.0;
    boost::shared_ptr<MemoryDataLayer<float> > memory_data_layer;
    memory_data_layer = boost::static_pointer_cast<MemoryDataLayer<float>>(net.layer_by_name("data"));

    //memory_data_layer->AddMatVector(imageVector, labelVector);
    //Datum datum;
    //ReadImageToDatum(imgName, 1, 256, 256, &datum);
	Datum datum;
	//cv::Mat cv_img;
	//cv::resize(image, cv_img, cv::Size(224, 224));
	CVMatToDatum(image,&datum);

    vector<Datum> datums;
    for (int i = 0; i < 1; i++)
        datums.push_back(datum);

    memory_data_layer->AddDatumVector(datums);
    const vector<Blob<float>*>& results = net.ForwardPrefilled(&loss);

	//show map
	const boost::shared_ptr<Blob<float> > feature_blob2 = net.blob_by_name("prob");
	int num2=feature_blob2->num();
	int channel2=feature_blob2->channels();
	int imrows2=feature_blob2->height();
	int imcols2=feature_blob2->width();
	cout<<"num:"<<num2<<" channel:"<<channel2<<" rows:"<<imrows2<<"; cols:"<<imcols2<<endl;
	Mat blob_data=Mat::zeros(imrows2,imcols2,CV_64FC1);
	const float* output=feature_blob2->cpu_data();
	int n=0;
	for (int c = 0; c < 1; ++c) {
		for (int h = 0; h < imrows2; ++h) {
			for (int w = 0; w < imcols2;++w) {
				blob_data.at<float>(h,w)=*(feature_blob2->mutable_cpu_data() + feature_blob2->offset(n,c,h,w));
			}
		}
	}
	float min=0,max=0;
	cv::minMaxIdx(blob_data,&min,&max);
	cout<<"min:"<<min<<"; max:"<<max<<endl;

	Mat feaImg2(Size(imrows2, imcols2), CV_8UC1);
	for (int row = 0; row < imrows2; row++) {
        for (int col = 0; col < imcols2; col++) {
            int d = row * imcols2 + col;
            feaImg2.at<uchar>(row,col) = floor(255.0*(blob_data.at<float>(row,col)-min)/(max-min));
        }
    }
    imshow("map", feaImg2);
	waitKey();
	//debug
	for (int row = 0; row < imrows2; row++) {
        for (int col = 0; col < imcols2; col++) {
			cout<<blob_data.at<float>(row,col)<<" ";
        }
		cout<<endl;
    }
	system("pause");
    return 0;
}
*/