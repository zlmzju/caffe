#include <string>
#include <vector>
#include <string>
#include "opencv2/core/core.hpp"

using namespace std;
using namespace cv;

class CNNFeature
{
public:
	// init the caffe model with prototxt file, useGPU_ID=-1 mean do not use any GPU
	CNNFeature( string protofile="../models/VGG_Small/param.prototxt", 
				string caffemodel="../models/VGG_Small/model", 
				int useGPU_ID=-1); 
	~CNNFeature();
	// given an image, extract and fill to the feature vector, return the dimension
	int ExtractFeature(Mat &image,float *feature, int maxSize=4096, string layerName="fc7");	
}; // class CNNFeature