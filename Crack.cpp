#include "CrackDetection.h"
#include <cstring>
#include <fstream>

using namespace std;
using namespace cv;
//using namespace boost::filesystem;

ofstream featureFile("Crack_feature.txt");

void logFeature(double *features, int len)
{
	for(int i=0;i<len; ++i)
	{
		featureFile << features[i] << "\t";
	}
	featureFile << endl;
}

int totalNum, cracked, nonCracked;


int work(string filename)
{
	//IplImage* src_RGB = cvLoadImage("sample.jpg");
	//IplImage* src_RGB = cvLoadImage("00Y_3.jpg");
	IplImage* src_RGB = cvLoadImage(filename.c_str());
	
	//cvShowImage("aa",src_RGB);
	CvSize size = cvGetSize(src_RGB);
	IplImage* src_gray = cvCreateImage(size, 8, 1);
	cvCvtColor(src_RGB,src_gray,CV_BGR2GRAY);

	double features[9];

	IplImage* src_R = cvCreateImage(cvSize(src_RGB->width, src_RGB->height), IPL_DEPTH_8U, 1);
	IplImage* src_G = cvCreateImage(cvSize(src_RGB->width, src_RGB->height), IPL_DEPTH_8U, 1);
	IplImage* src_B = cvCreateImage(cvSize(src_RGB->width, src_RGB->height), IPL_DEPTH_8U, 1);

	cvSplit(src_RGB,src_B,src_G,src_R,NULL);

	CrackDetection::GetFeatures(src_B,features);
	logFeature(features, 9);

	cvReleaseImage(&src_gray);
	//cvReleaseImage(&src_RGB);
	cvReleaseImage(&src_R);
	cvReleaseImage(&src_G);
	cvReleaseImage(&src_B);

	//SVMClassifer svm;
	//SVMFeature svmfeature;
	//vector<double> vtFeature(features,features+9);
	//svmfeature.SetFeature(vtFeature);
	//String strFileName = "Crack_train_B.model";
	//svm.SetFeature(&svmfeature);
	//int nRt = svm.LoadModel(strFileName.c_str());
	//double RESULT = svm.Predict();
	//cout << RESULT << endl;

	//------------------------------------------------------------//
	SVM svm;
	//svm.load("crack_nonLinear.model");
	svm.load("crack_linear_120sample.model");

	Mat sampleMat = Mat(1, 9, CV_64FC1, features);
	sampleMat.convertTo(sampleMat, CV_32FC1);

	int res = svm.predict(sampleMat);
	
	ofstream outfile("Crack.txt");
	if(res==1)
	{
		outfile << "1 ÓÐÁÑÎÆ" << endl;
		cout << "1 ÓÐÁÑÎÆ" << endl;
		cracked++;
	}
	else if(res==-1)
	{
		outfile << "-1 ÎÞÁÑÎÆ" << endl;
		cout << "-1 ÎÞÁÑÎÆ" << endl;
		nonCracked++;
	}	
	else
		cout << "fuck" << endl;
	//------------------------------------------------------------//

	return 0;
}


int main(int argc, char* argv[])
{
	string filename("");

	if(argc == 1)
	{
		filename = "T1.jpg";
	}
	else if(argc == 2)
	{
		filename = string(argv[1]);
	}
	else
	{
		cout << "ÇëÊäÈëºÏ·¨Í¼Æ¬Â·¾¶" << endl;
		return -1;
	}

	cout << "¶ÁÈ¡Í¼Æ¬£º " << filename << endl;
	Mat testImg=imread(filename,CV_LOAD_IMAGE_GRAYSCALE);
	
	//namedWindow("aaa");
	//imshow("aaa", testImg);
	//waitKey(0);

	if(!testImg.data)
	{
		cout << "¶ÁÈ¡Í¼Æ¬Ê§°Ü£¡" << endl;
		return -1;
	}
		
	work(filename);

	//path p("C:\\Users\\kidd\\Documents\\Visual Studio 2013\\Projects\\selectFiles\\selectFiles\\smallSize");
	//path p("C:\\Users\\kidd\\Documents\\Visual Studio 2013\\Projects\\selectFiles\\backgroundToBlack\\chihen_black\\ÎÞ");



	//path p("liewen_black\\test_new\\");
	//directory_iterator end;
	//for(directory_iterator iter(p);
	//	iter!=end; ++iter)
	//{
	//	filename = iter->path().string(); 
	//	cout << filename << endl;
	//	work(filename);
	//}

	//cout << "ÁÑÎÆÊýÁ¿£º" << cracked << "\tÎÞÁÑÎÆÊýÁ¿" << nonCracked << endl;
	//cout << "ÁÑÎÆ±ÈÀý£º" << (double)cracked/(cracked+nonCracked) << "·ÇÁÑÎÆ±ÈÀý£º "<< (double)nonCracked/(cracked+nonCracked) << endl;
}


