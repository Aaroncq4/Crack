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
	static int IsCrackTongue(const IplImage* src,int class_sign,const double radius_ratio_thres = 1.0/15.0,double thres_1 = 0.10,double thres_2 = 2.0);//供外部代码调用的识别程序
	static int IsCrackTongue(const IplImage* src,const char* dir,const double radius_ratio_thres = 1.0/15.0,double thres_1 = 0.10,double thres_2 = 2.0);//带文件操作的函数重载
	static int IsCrackTongue(const IplImage* src,int class_sign,int index,const double radius_ratio_thres,double thres_1,double thres_2);//用于分析delta(thres_1)以及p(radius_ratio_thres)的函数重载
	static int IsCrackTongue(const IplImage* src,const double radius_ratio_thres = 1.0/15.0,double thres_1 = 0.10,double thres_2 = 2.0);//用于主程序调用的识别函数
	static void runtime (int t);
	static void GetFeatures(const IplImage* src,double features [9],const double radius_ratio_thres = 1.0/15.0,double thres_1 = 0.10,double thres_2 = 2.0);//用于项目程序调用的特征提取函数
private:
	static IplImage* ImageResize(const IplImage* src,int width);//调整样本分辨率
	static IplImage* DeleteBlack(IplImage* src);//去除多余黑色背景
	static double sech(double input);//双曲正切函数
	static IplImage* ImageCrop(IplImage* src,double ratio = Ratio);//获取待测区域
	static void InverseGaussianMask(int radius,double* addr);//计算反高斯掩膜
	static double Get_t(const IplImage* src);//求图像非零元素标准差t
	static double Get_L(int x0,int y0,int radius,double* IGM,const IplImage *src,double t,double thres = 0.10,double g = 0.5);//求(x0,y0)点反高斯掩膜值
	static void IGMImage(const IplImage* src,IplImage* dst,double* IGM,int radius,double t,double thres = 0.10,double g = 0.5);//得到经过反高斯掩膜处理后的图像（待优化）
	static double otsu(const IplImage *frame);//获取二值化阈值
	static void Histeq(const IplImage* src,IplImage* dst);//直方图均衡化
	static void cvThresholdOtsu(IplImage* src, IplImage* dst);//利用Otsu阈值进行图像二值化
	static void KillDot(const IplImage* src,IplImage* dst,double thres);//去除线度低于阈值thres的块(只接受8位图像)
	static void KillDot(const IplImage* src,IplImage* dst,double thres,double* features);//去除线度低于阈值thres的块并提取特征
	static double KD(uchar* src,uchar* pflag,int rownum,int colnum,int row,int col,int width_step);//KillDot的核心函数
	static double KD(uchar* src,uchar* dst,int rownum,int colnum,int row,int col,int width_step,double* features);//KillDot的核心函数重载，可用于提取相关特征
	static void Write_file(double* features,int len,int class_sign);//写入特征文本文件
	static void Write_file(double* features,int len,int class_sign,int index);
	static int min_pos(double* input,int len);//计算数组最小元素位置
	static double cmp(const double &a,const double &b);//用于快速排序的比较函数
};

