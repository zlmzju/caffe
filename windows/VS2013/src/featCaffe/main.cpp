#include <stdio.h>
#include "CNNFeature.h"
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
using namespace cv;
int main()
{
	CNNFeature cnnFeature;
	string imgName = "./examples/images/cat.jpg";
    Mat image = imread(imgName);
	int maxSize=4096;
	//test 1
	float *feature=(float*)malloc(sizeof(float)*maxSize);
	int dim=cnnFeature.ExtractFeature(image,feature,maxSize);
	printf("\n");
	for(int i=0;i<10;i++)
	{
		printf("%4.2f\t",feature[i]);
	}
	free(feature);

	//test 2
	float *feature2=(float*)malloc(sizeof(float)*maxSize);
	int dim2=cnnFeature.ExtractFeature(image,feature2,maxSize);
	printf("\n");
	for(int i=0;i<10;i++)
	{
		printf("%4.2f\t",feature2[i]);
	}
	free(feature2);
	printf("\n");
	system("pause");
	return 0;
}