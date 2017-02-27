#pragma once

#include "opencv2/opencv.hpp"
#include "opencv/cv.h"
#include <iostream>
#include <fstream>
#include <iomanip>
#include <stdlib.h>
#include <cstring> 
#include <math.h>
#include <algorithm>
#include <time.h>

#include <boost\filesystem\path.hpp>
#include <boost\filesystem\operations.hpp>


#define Ratio 0.6
#define GrayScale 256
#define MAX_LEN 1000

//#define PI 3.14159265
#define e 2.718281828459

using namespace cv;
using namespace std;

class CrackDetection
{
public:
	CrackDetection(void);
	~CrackDetection(void);
	static int IsCrackTongue(const IplImage* src,int class_sign,const double radius_ratio_thres = 1.0/15.0,double thres_1 = 0.10,double thres_2 = 2.0);//���ⲿ������õ�ʶ�����
	static int IsCrackTongue(const IplImage* src,const char* dir,const double radius_ratio_thres = 1.0/15.0,double thres_1 = 0.10,double thres_2 = 2.0);//���ļ������ĺ�������
	static int IsCrackTongue(const IplImage* src,int class_sign,int index,const double radius_ratio_thres,double thres_1,double thres_2);//���ڷ���delta(thres_1)�Լ�p(radius_ratio_thres)�ĺ�������
	static int IsCrackTongue(const IplImage* src,const double radius_ratio_thres = 1.0/15.0,double thres_1 = 0.10,double thres_2 = 2.0);//������������õ�ʶ����
	static void runtime (int t);
	static void GetFeatures(const IplImage* src,double features [9],const double radius_ratio_thres = 1.0/15.0,double thres_1 = 0.10,double thres_2 = 2.0);//������Ŀ������õ�������ȡ����
private:
	static IplImage* ImageResize(const IplImage* src,int width);//���������ֱ���
	static IplImage* DeleteBlack(IplImage* src);//ȥ�������ɫ����
	static double sech(double input);//˫�����к���
	static IplImage* ImageCrop(IplImage* src,double ratio = Ratio);//��ȡ��������
	static void InverseGaussianMask(int radius,double* addr);//���㷴��˹��Ĥ
	static double Get_t(const IplImage* src);//��ͼ�����Ԫ�ر�׼��t
	static double Get_L(int x0,int y0,int radius,double* IGM,const IplImage *src,double t,double thres = 0.10,double g = 0.5);//��(x0,y0)�㷴��˹��Ĥֵ
	static void IGMImage(const IplImage* src,IplImage* dst,double* IGM,int radius,double t,double thres = 0.10,double g = 0.5);//�õ���������˹��Ĥ������ͼ�񣨴��Ż���
	static double otsu(const IplImage *frame);//��ȡ��ֵ����ֵ
	static void Histeq(const IplImage* src,IplImage* dst);//ֱ��ͼ���⻯
	static void cvThresholdOtsu(IplImage* src, IplImage* dst);//����Otsu��ֵ����ͼ���ֵ��
	static void KillDot(const IplImage* src,IplImage* dst,double thres);//ȥ���߶ȵ�����ֵthres�Ŀ�(ֻ����8λͼ��)
	static void KillDot(const IplImage* src,IplImage* dst,double thres,double* features);//ȥ���߶ȵ�����ֵthres�Ŀ鲢��ȡ����
	static double KD(uchar* src,uchar* pflag,int rownum,int colnum,int row,int col,int width_step);//KillDot�ĺ��ĺ���
	static double KD(uchar* src,uchar* dst,int rownum,int colnum,int row,int col,int width_step,double* features);//KillDot�ĺ��ĺ������أ���������ȡ�������
	static void Write_file(double* features,int len,int class_sign);//д�������ı��ļ�
	static void Write_file(double* features,int len,int class_sign,int index);
	static int min_pos(double* input,int len);//����������СԪ��λ��
	static double cmp(const double &a,const double &b);//���ڿ�������ıȽϺ���
};

