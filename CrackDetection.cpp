#include "CrackDetection.h"


using namespace cv;
using namespace std;

//
void showImage(IplImage *img, string windowName)
{
	cvNamedWindow(windowName.c_str());
	cvShowImage(windowName.c_str(), img);
	cvWaitKey(0);
}
//

CrackDetection::CrackDetection(void)//���캯��
{
}

CrackDetection::~CrackDetection(void)//��������
{
}

void CrackDetection::runtime (int t)
{
	t = abs(t);
	int n = t/CLOCKS_PER_SEC;
	int r = t-CLOCKS_PER_SEC*n;
	cout << endl;
	cout << "����ʱ�䣺";
	cout << n << " �� " << r << " ����" << endl;
}

IplImage* CrackDetection::ImageResize(const IplImage* src,const int width)
{
	int src_height = src->height;
	int src_width = src->width;
	double ratio = (double)src_height / (double)src_width;
	int height = (int)(width * ratio + 0.5);
	IplImage* dst = cvCreateImage(cvSize(width,height),src->depth,src->nChannels);
	cvResize(src,dst,CV_INTER_LINEAR);  
	return dst;
	//cvNamedWindow("resize");
	//cvShowImage("resize",dst);
	//cvWaitKey();
}

IplImage* CrackDetection::DeleteBlack(IplImage* src)
{
	//X=imread(InputImage);
	//[row,col,cha]=size(X);
	int width = src->width;
	int height = src->height;
	uchar* addr;
	int top = 65535,bottom = 65535,left = 65535,right = 65535;

	for(int i = 0;i < height;i++)
	{
		addr = (uchar*)(src->imageData + src->widthStep*i);
		for(int j = 0;j < width;j++)
		{
			if ((int)addr[j] > 100)
			{
				//cout << "pix at(" << i << "," << j << ") has value: " << (int)addr[j] << endl; 

				top = i;
				break;
			}
		}
		if(top == i)
			break;
	}

	for(int i = height - 1;i >= 0;i--)
	{
		addr = (uchar*)(src->imageData + src->widthStep*i);
		for(int j = 0;j < width;j++)
		{
			if ((int)addr[j] > 100)
			{
				//cout << "pix at(" << i << "," << j << ") has value: " << (int)addr[j] << endl;

				bottom = i;
				break;
			}
		}
		if(bottom == i)
			break;
	}

	for(int i = 0;i < width;i++)
	{
		for(int j = 0;j < height;j++)
		{
			addr = (uchar*)(src->imageData + src->widthStep*j);
			if ((int)addr[i] > 100)
			{
				//cout << "pix at(" << i << "," << j << ") has value: " << (int)addr[j] << endl;

				left = i;
				break;
			}
		}
		if(left == i)
			break;
	}	

	for(int i = width - 1;i >= 0;i--)
	{
		for(int j = 0;j < height;j++)
		{
			addr = (uchar*)(src->imageData + src->widthStep*j);
			if ((int)addr[i] > 100)
			{
				//cout << "pix at(" << i << "," << j << ") has value: " << (int)addr[j] << endl;

				right = i;
				break;
			}
		}
		if(right == i)
			break;
	}
	//cout << left << "\t" << right << "\t" 
	//	<< top << "\t" << bottom << endl;

	cvSetImageROI(src , cvRect(left,top,right - left,bottom - top));
	IplImage* dst = cvCreateImage(cvSize(right - left,bottom - top),src->depth,src->nChannels);
	cvCopy(src,dst,0);
	cvResetImageROI(src);
	return dst;
}

double CrackDetection::cmp(const double &a,const double &b)
{
	return a > b;
}

double CrackDetection::sech(const double input)
{
	double output;
	output = 2.0/(exp(input)+exp(-input));
	//output = 2.0/(1+1<<abs((int)input));
	return output;
}

IplImage* CrackDetection::ImageCrop(IplImage* src,double ratio)
{
	int width = (int)((double)(src->width)*ratio + 0.5);
	int height = (int)((double)(src->height)*ratio + 0.5);
	int x = (int)((double)(src->width)*(1 - ratio)/2 + 0.5);
	int y = (int)((double)(src->height)*(1 - ratio)/2 + 0.5);
	IplImage* dst = cvCreateImage(cvSize(width+1,height+1),src->depth,src->nChannels);
	cvSetImageROI(src , cvRect(x,y,width+1,height+1));
	cvCopy(src,dst,0);
	cvResetImageROI(src);
	return dst;
}

void CrackDetection::InverseGaussianMask(int radius,double* addr)//���㷴��˹��Ĥ
{
	if (radius < 1)
	{
		cout<<"��Ĥ�뾶С��1��";
		return;
	}

	
	int temp_1;
	double A = 0,B = 0,C = 0,D = 0;

	//������Ĥ
	for(int i = 1;i<=2*radius+1;i++)
	{
		for(int j = 1;j<=2*radius+1;j++)
		{
			A = exp((double)(0-(double)((i-radius-1)*(i-radius-1)+(j-radius-1)*(j-radius-1))/(double)(2*radius*radius)));
			C += A;
		}
	}

	D = (2*radius+1)*(2*radius+1) - C;

	for(int i = 1;i<=2*radius+1;i++)
	{
		for(int j = 1;j<=2*radius+1;j++)
		{
			A = exp((double)(0-(double)((i-radius-1)*(i-radius-1)+(j-radius-1)*(j-radius-1))/(double)(2*radius*radius)));
			B = 1 - A;
			if (D != 0)
			{
				temp_1 = (int)((i-1)*(2*radius+1)+j-1);
				addr[temp_1] = B/D;
			}
		}
	}
}

double CrackDetection::Get_t(const IplImage* src)//��ͼ�����Ԫ�ر�׼��t
{
	if (!src)
	{
		cout<<"ͼ��ָ��Ϊ�գ�";
		return 0;
	}

	uchar* addr;

	long NUM = cvCountNonZero(src);
	long long SUM = 0;
	double AVG,SDV = 0;
	Mat temp = Mat(src);

	//cout<<src->height<<' '<<src->width<<endl;
	//cout<<temp.rows<<' '<<temp.cols<<endl;


	CvScalar S;
	S = cvSum(src);//���
	//cout<<S.val[0]<<endl;


	for(int i = 0;i < temp.rows;i++)
	{
		for(int j = 0;j < temp.cols;j++)
		{
			addr = (uchar*)(temp.data+i*temp.step[0]+j*temp.step[1]);
			SUM += (int)addr[0];
			//cout<<(int)addr[0]<<' ';
		}
	}

	//cout<<SUM<<endl;

	if (NUM > 0)
		AVG = (double)SUM/(double)NUM;
	else
		return 0;

	//cout<<AVG<<endl;

	for(int i = 0;i < temp.rows;i++)
	{
		for(int j = 0;j < temp.cols;j++)
		{
			addr = (uchar*)(temp.data+i*temp.step[0]+j*temp.step[1]);
			SDV += ((double)((int)addr[0]) - AVG)*((double)((int)addr[0]) - AVG);
		}
	}

	SDV = sqrt(SDV/(double)(NUM - 1));

	addr = NULL;

	temp.release();

	cout<<"SDV"<<" = "<<SDV<<endl;

	return SDV;
}

double CrackDetection::Get_L(int x0,int y0,int radius,double* IGM,const IplImage *src,double t,double thres,double g)//��ȡһ���Lֵ,x0��y0���Ϊ(0,0)
{
	//Mat temp(src);
	int width = 2*radius + 1;
	double sech_list[GrayScale] = {0};
	double sech_flag[GrayScale] = {0};
	double m = 0,L;
	uchar* addr = (uchar*)(src->imageData + src->widthStep*y0 + x0);
	//double *addr;
	int A = (int)(addr[0]);
	for (int k = 0;k < width;k++)
	{
		for (int l = 0;l < width;l++)
		{
			addr = (uchar*)(src->imageData + src->widthStep*(y0-radius+k) + x0-radius+l);
			if (A < (int)(addr[0]))
			{
				if (sech_flag[(int)(addr[0])-A] == 1)
				{
					m += IGM[k*width+l]*sech_list[(int)(addr[0])-A];
				}
				else
				{
					if(t == 0)
					{
						cout<<"2��"<<endl;
					}
					sech_list[(int)(addr[0])-A] = pow(sech(((double)((int)(addr[0])-(A)))/t),5);
					//sech_list[(int)(addr[0])-A] = sech(((double)((int)(addr[0])-(A)))/t);
					sech_flag[(int)(addr[0])-A] = 1;
					m += IGM[k*width+l]*sech_list[(int)(addr[0])-A];
				}
			}
			else
				m +=IGM[k*width+l];
		}

	}

	if (m < g - thres)
		L = g - m;
	else
		L = 0;
	addr = NULL;
	return L;
}

void CrackDetection::IGMImage(const IplImage* src,IplImage* dst,double* IGM,int radius,double t,double thres,double g)
{
	if (!src)
	{
		cout<<"ͼ��ָ��Ϊ�գ�";
		return;
	}
	//Mat temp(src);
	for (int i = radius;i < src->height - radius;i++)
	{
		for (int j = radius;j < src->width - radius;j++)
		{
			((uchar*)(dst->imageData + i*dst->widthStep))[j] = (int)(255.0*Get_L(j,i,radius,IGM,src,t,thres)+0.5);
		}
	}


	//	temp.release();
	//return output;
}

double CrackDetection::otsu(const IplImage *frame) //�������ֵ
{
	//frame�Ҷȼ�
	int pixelCount[GrayScale] = {0};
	double pixelPro[GrayScale] = {0};
	int i, j, pixelSum = frame->width * frame->height, threshold = 0;
	uchar* data = (uchar*)frame->imageData;
	double w0, w1, u0tmp, u1tmp, u0, u1, deltaTmp, deltaMax = 0;
	//ͳ��ÿ���Ҷȼ������صĸ���
	for(i = 0; i < frame->height; i++)
	{
		for(j = 0;j < frame->width;j++)
		{
			pixelCount[(int)data[i * frame->width + j]]++;
		}
	}

	//����ÿ���Ҷȼ���������Ŀռ����ͼ��ı���
	for(i = 0; i < GrayScale; i++)
	{
		pixelPro[i] = (double)pixelCount[i] / pixelSum;
	}

	//�����Ҷȼ�[0,255],Ѱ�Һ��ʵ�threshold

	for(i = 0; i < GrayScale; i++)
	{
		w0 = w1 = u0tmp = u1tmp = u0 = u1 = deltaTmp = 0;
		for(j = 0; j < GrayScale; j++)
		{
			if(j <= i)   //��������
			{
				w0 += pixelPro[j];
				u0tmp += j * pixelPro[j];
			}
			else   //ǰ������
			{
				w1 += pixelPro[j];
				u1tmp += j * pixelPro[j];
			}
		}
		u0 = u0tmp / w0;
		u1 = u1tmp / w1;
		deltaTmp = (double)(w0 *w1* pow((u0 - u1), 2)) ;
		if(deltaTmp > deltaMax)
		{
			deltaMax = deltaTmp;
			threshold = i;
		}
	}
	return (double)threshold;
}

void CrackDetection::Histeq(const IplImage* src,IplImage* dst)
{
	if (!src||!dst||src->height!=dst->height||src->width!=dst->width)
	{
		cout<<"ֱ��ͼ���⻯�����������";
		return;
	}
	uchar* addr;
	int height = src->height;
	int width = src->width;
	double hist[GrayScale] = {0};


	for(int i=0; i<height; i++)
	{
		addr=(uchar*)(src->imageData+src->widthStep*i);
		for(int j=0; j<width; j++)
		{
			hist[addr[j]]++;
		}
	}

	
	//normalize histogram
	long size = width * height;

	for(int i=1; i<256; i++)
	{
		hist[i] = hist[i] + hist[i - 1];
		//cout<<hist[i]<<' ';
	}

	for(int i=0; i<256; i++)
	{
		hist[i]=hist[i]/(double)size;
		//cout<<hist[i]<<' ';
	}

	for (int i = 0;i < height;i++)
	{
		addr = (uchar*)(src->imageData+src->widthStep*i);
		for (int j = 0;j < width;j++)
		{
			((uchar*)(dst->imageData + dst->widthStep*i))[j] = (int)((double)(GrayScale - 1)*hist[addr[j]]);
		}
	}
	addr = NULL;
}

void CrackDetection::cvThresholdOtsu(IplImage* src, IplImage* dst)
{
	int height=src->height;
	int width=src->width;

	//histogram
	double histogram[256]= {0};
	for(int i=0; i<height; i++)
	{
		unsigned char* p=(unsigned char*)src->imageData+src->widthStep*i;
		for(int j=0; j<width; j++)
		{
			histogram[*p++]++;
		}
	}
	//normalize histogram
	int size=height*width;
	for(int i=0; i<256; i++)
	{
		histogram[i]=histogram[i]/size;
	}

	//average pixel value
	double avgValue=0;
	for(int i=0; i<256; i++)
	{
		avgValue+=i*histogram[i];
	}

	int threshold;
	double maxVariance=0;
	double w=0,u=0;
	for(int i=0; i<256; i++)
	{
		w+=histogram[i];
		u+=i*histogram[i];

		double t=avgValue*w-u;
		double variance=t*t/(w*(1-w));
		if(variance>maxVariance)
		{
			maxVariance=variance;
			threshold=i;
		}
	}

	cvThreshold(src,dst,threshold,255,CV_THRESH_BINARY);
}

double CrackDetection::KD(uchar* src,uchar* dst,int rownum,int colnum,int row,int col,int width_step,double* fat_est)
{
	//��ʼ������
	double MAX_distance = 0;

	double temp = 0;

	//�˸����������ֵ
	int nDx[]={-1,0,1,1,1,0,-1,-1};
	int nDy[]={-1,-1,-1,0,1,1,1,0};

	int nStart ;
	int nEnd   ;
	int nSeedX, nSeedY;

	// ����������ʼ��
	nSeedX = colnum;
	nSeedY = rownum;
	dst[nSeedY*width_step+nSeedX] = 255;
	//��ʼ��
	nStart = 0 ;
	nEnd   = 0 ;
	short* XQue ;
	short* YQue ;

	// ����ռ�
	XQue = new short [row*col];
	YQue = new short [row*col];


	XQue[nEnd] = nSeedX;
	YQue[nEnd] = nSeedY;

	//���嵱ǰ�������
	int nCurrX ;
	int nCurrY ;


	int xx;
	int yy;

	while (nStart<=nEnd)
	{
		// ��ǰ�������
		nCurrX = XQue[nStart];
		nCurrY = YQue[nStart]; 


		// �Ե�ǰ��������8������б���
		for (int k=0; k<8; k++)
		{

			xx = nCurrX+nDx[k];
			yy = nCurrY+nDy[k];

			if ( (xx < col) && (xx>=0) && (yy<row) && (yy>=0)
				&& (dst[yy*width_step+xx]==0)  && ((int)src[yy*width_step+xx] == (int)src[nCurrY*width_step+nCurrX]))
			{
				// ��ջ��β��ָ�����һλ
				nEnd++;

				// ����(xx��yy) ѹ��ջ
				XQue[nEnd] = xx;
				YQue[nEnd] = yy;

				for (int i = 0;i <= nEnd;i++)
				{
					temp = sqrt((double)((XQue[nEnd]-XQue[i])*(XQue[nEnd]-XQue[i]) + (YQue[nEnd]-YQue[i])*(YQue[nEnd]-YQue[i])));
					if(MAX_distance < temp)
						MAX_distance = temp;
				}

				//��dstͼ�����Ӧ�����
				dst[yy*width_step+xx] = 255;
			}
		}
		nStart++;
	}

	*fat_est = MAX_distance/(nEnd+1);//��ȡ�����ߵ�������Ϣ

	// �ͷ��ڴ�
	delete []XQue;
	delete []YQue;
	XQue = NULL ;
	YQue = NULL ;
	return MAX_distance;
}

double CrackDetection::KD(uchar* src,uchar* dst,int rownum,int colnum,int row,int col,int width_step)
{
	//��ʼ������
	double distance = 0;

	double temp = 0;

	//�˸����������ֵ
	int nDx[]={-1,0,1,1,1,0,-1,-1};
	int nDy[]={-1,-1,-1,0,1,1,1,0};

	int nStart ;
	int nEnd   ;
	int nSeedX, nSeedY;

	// ����������ʼ��
	nSeedX = colnum;
	nSeedY = rownum;
	dst[nSeedY*width_step+nSeedX] = 255;
	//��ʼ��
	nStart = 0 ;
	nEnd   = 0 ;
	short* XQue ;
	short* YQue ;

	// ����ռ�
	XQue = new short [row*col];
	YQue = new short [row*col];


	XQue[nEnd] = nSeedX;
	YQue[nEnd] = nSeedY;

	//���嵱ǰ�������
	int nCurrX ;
	int nCurrY ;


	int xx;
	int yy;

	while (nStart<=nEnd)
	{
		// ��ǰ�������
		nCurrX = XQue[nStart];
		nCurrY = YQue[nStart]; 


		// �Ե�ǰ��������8������б���
		for (int k=0; k<8; k++)
		{

			xx = nCurrX+nDx[k];
			yy = nCurrY+nDy[k];

			if ( (xx < col) && (xx>=0) && (yy<row) && (yy>=0)
				&& (dst[yy*width_step+xx]==0)  && ((int)src[yy*width_step+xx] == (int)src[nCurrY*width_step+nCurrX]))
			{
				// ��ջ��β��ָ�����һλ
				nEnd++;

				// ����(xx��yy) ѹ��ջ
				XQue[nEnd] = xx;
				YQue[nEnd] = yy;

				for (int i = 0;i <= nEnd;i++)
				{
					temp = sqrt((double)((XQue[nEnd]-XQue[i])*(XQue[nEnd]-XQue[i]) + (YQue[nEnd]-YQue[i])*(YQue[nEnd]-YQue[i])));
					if(distance < temp)
						distance = temp;
				}

				//��dstͼ�����Ӧ�����
				dst[yy*width_step+xx] = 255;
			}
		}
		nStart++;
	}

	// �ͷ��ڴ�
	delete []XQue;
	delete []YQue;
	XQue = NULL ;
	YQue = NULL ;
	return distance;
}

int CrackDetection::min_pos(double* input,int len)
{
	if(len == 1)
	{
		return 0;
	}

	double temp = input[0];
	int Pos = 0;

	for(int i = 1;i < len;i++)
	{
		if(input[i] < temp)
		{
			Pos = i;
			temp = input[i];
		}
	}

	return Pos;
}

void CrackDetection::KillDot(const IplImage* src,IplImage* dst,double thres,double* features)
{
	cvSetZero(dst);

	const int LEN = 3;

	int row = src->height;
	int col = src->width;
	int width_step = src->widthStep;
	double fat_est;
	double MAX_distance;
	double MAX_3_fat_est_list[LEN];
	double MAX_3_distance_list[LEN];
	double* distance_list = new double [row*col];
	double* fat_est_list = new double [row*col];
	int counter = 0;

	memset(MAX_3_fat_est_list,0,sizeof(MAX_3_fat_est_list));
	memset(MAX_3_distance_list,0,sizeof(MAX_3_distance_list));
	memset(distance_list,0,sizeof(distance_list));
	memset(fat_est_list,0,sizeof(fat_est_list));

	//ÿ���߶ȼ�������ͼ������

	IplImage* temp_1 = cvCreateImage(cvSize(src->width,src->height),src->depth,src->nChannels);
	cvCopy(src,temp_1);

	//ÿ���߶ȼ������ͼ������

	IplImage* temp_2 = cvCreateImage(cvSize(src->width,src->height),src->depth,src->nChannels);
	cvSetZero(temp_2);

	for(int i = 0;i < row;i++)
	{
		for(int j = 0;j < col;j++)
		{
			if((int)(((uchar*)(temp_1->imageData + i*temp_1->widthStep))[j]) == 0)
			{
				continue;
			}
			else
			{
				uchar* src_addr = (uchar*)(temp_1->imageData);
				uchar* dst_addr = (uchar*)(temp_2->imageData);
				MAX_distance = KD(src_addr,dst_addr,i,j,row,col,width_step,&fat_est);
				distance_list[counter] = MAX_distance;
				fat_est_list[counter] = fat_est;
				//cout<<fat_est<<endl;
				counter++;
				int Min_pos = min_pos(MAX_3_distance_list,LEN);

				//cout<<Min_pos<<endl;

				if(MAX_3_distance_list[Min_pos] < MAX_distance)
				{
					MAX_3_distance_list[Min_pos] = MAX_distance;
					MAX_3_fat_est_list[Min_pos] = fat_est;
				}

				if (MAX_distance > thres)
					cvAdd(dst,temp_2,dst);
				cvSub(temp_1,temp_2,temp_1);
				cvSetZero(temp_2);
			}
		}
	}

	sort(MAX_3_distance_list,MAX_3_distance_list+3);

	for(int i = 1;i < LEN + 1;i++)
	{
		features[i] = MAX_3_distance_list[i-1];
		//features[i+1] = MAX_3_fat_est_list[i];
		//cout<<MAX_3_distance_list[i]<<"	"<<MAX_3_fat_est_list[i]<<endl;
	}

	for(int i = LEN + 1;i < 2*LEN + 1;i++)
	{
		features[i] = MAX_3_fat_est_list[i - LEN - 1];
	}

	double dis_sum = 0;
	double fat_sum = 0;


	for(int i = 0;i < counter;i++)
	{
		dis_sum += distance_list[i];
		fat_sum += fat_est_list[i];
	}
	
	double dis_avg = dis_sum/counter;
	double fat_avg = fat_sum/counter;

	double dis_std_dev = 0;
	double fat_std_dev = 0;

	for(int i = 0;i < counter;i++)
	{
		dis_std_dev += (distance_list[i] - dis_avg)*(distance_list[i] - dis_avg);
		fat_std_dev += (fat_est_list[i] - fat_avg)*(fat_est_list[i] - fat_avg);
	}

	dis_std_dev = sqrt(dis_std_dev/counter);
	fat_std_dev = sqrt(fat_std_dev/counter);

	features[2*LEN + 1] = dis_std_dev;
	//features[2*LEN + 1] = dis_std_dev;
	features[2*(LEN + 1)] = fat_std_dev;

	//cout<<fat_std_dev<<endl;

	delete []fat_est_list;
	fat_est_list = NULL;
	delete []distance_list;
	distance_list = NULL;
	cvReleaseImage(&temp_1);
	cvReleaseImage(&temp_2);
}

void CrackDetection::KillDot(const IplImage* src,IplImage* dst,double thres)
{
	cvSetZero(dst);
	//cvCopy(src,dst);
	int row = src->height;
	int col = src->width;
	int width_step = src->widthStep;
	double MAX_distance;

	//ÿ���߶ȼ�������ͼ������

	IplImage* temp_1 = cvCreateImage(cvSize(src->width,src->height),src->depth,src->nChannels);
	cvCopy(src,temp_1);

	//ÿ���߶ȼ������ͼ������

	IplImage* temp_2 = cvCreateImage(cvSize(src->width,src->height),src->depth,src->nChannels);
	cvSetZero(temp_2);

	for(int i = 0;i < row;i++)
	{
		for(int j = 0;j < col;j++)
		{
			if((int)(((uchar*)(temp_1->imageData + i*temp_1->widthStep))[j]) == 0)
			{
				continue;
			}
			else
			{
				uchar* src_addr = (uchar*)(temp_1->imageData);
				uchar* dst_addr = (uchar*)(temp_2->imageData);
				MAX_distance = KD(src_addr,dst_addr,i,j,row,col,width_step);
				if (MAX_distance > thres)
					cvAdd(dst,temp_2,dst);
				cvSub(temp_1,temp_2,temp_1);
				cvSetZero(temp_2);
			}
		}
	}

	cvReleaseImage(&temp_1);
	cvReleaseImage(&temp_2);
}

void CrackDetection::Write_file(double* features,int len,int class_sign)
{
	ofstream outfile;
	outfile.open("..\\Crack_train.txt",ios::app);
	outfile<<class_sign<<" ";
	for(int i = 0;i < len - 1;i++)
	{
		outfile<<i+1<<':'<<features[i]<<" ";
	}
	outfile<<len<<':'<<features[len - 1]<<endl;
	outfile.close();
}

void CrackDetection::Write_file(double* features,int len,int class_sign,int index)
{
	ofstream outfile;
	char number[4];
	char CATA[100] = "..\\Crack_train_final";
	itoa(index,number,10);
	strcat(CATA,number);
	strcat(CATA,".txt");
	outfile.open(CATA,ios::app);
	outfile<<class_sign<<" ";
	for(int i = 0;i < len - 1;i++)
	{
		outfile<<features[i]<<" ";
	}
	outfile<<features[len - 1]<<endl;
	outfile.close();
}

int CrackDetection::IsCrackTongue(const IplImage* src,int class_sign,const double radius_ratio_thres,double thres_1,double thres_2)
{
	const int FEATURE_NUM = 9;
    double features[FEATURE_NUM];

	//��������ͼ�񸱱�
	IplImage* input = cvCreateImage(cvSize(src->width,src->height),src->depth,src->nChannels);
	cvCopy(src,input);

	//ȥ�������ɫ����
	IplImage* dst_1 = DeleteBlack(input);
	IplImage* dst_2 = ImageResize(dst_1,300);

	//����ȡ�����ֵĿ�Ⱥ͸߶�
	//int width = (int)((double)(src->width)*Ratio + 0.5);
	//int height = (int)((double)(src->height)*Ratio + 0.5);

	//�ü�ͼ����ȡ���о����󲿷�
	IplImage* cropped_image = ImageCrop(dst_2);

	//������Ĥ�뾶
	int radius = ((int)(cropped_image->height) > (int)(cropped_image->width))? ((int)(((double)(cropped_image->height))*radius_ratio_thres)):((int)(((double)(cropped_image->width))*radius_ratio_thres));
	
	//�뾶����
	features[0] = radius;

	//������Ĥ��������
	double* IGM = new double[(2*radius+1)*(2*radius+1)];

	//���㷴��˹��Ĥ����
	InverseGaussianMask(radius,IGM);

	//��ʾ����˹��Ĥ����
	//for(int i = 0;i<2*radius+1;i++)
	//{
	//	for (int j = 0;j<2*radius+1;j++)
	//	{
	//		cout<<IGM[i*(2*radius+1)+j]<<' ';
	//	}
	//	cout<<endl;
	//}
	
	//����tֵ
	double Ttest = Get_t(cropped_image);

	//���ڴ�ŷ���˹��Ĥ�������ͼ��
	IplImage* temp_1 = cvCreateImage(cvSize(cropped_image->width,cropped_image->height),cropped_image->depth,cropped_image->nChannels);
	cvSetZero(temp_1);

	//���ڴ��temp_1ֱ��ͼ���⻯���ͼ��
	IplImage* temp_2 = cvCreateImage(cvSize(cropped_image->width,cropped_image->height),cropped_image->depth,cropped_image->nChannels);
	cvSetZero(temp_2);

	//���ڴ��temp_2�Ķ�ֵ��ͼ��
	IplImage* temp_3 = cvCreateImage(cvSize(cropped_image->width,cropped_image->height),cropped_image->depth,cropped_image->nChannels);
	cvSetZero(temp_3);

	//���ڴ�����Ľ��ͼ
	IplImage* KDImage = cvCreateImage(cvSize(cropped_image->width,cropped_image->height),cropped_image->depth,cropped_image->nChannels);
	cvSetZero(KDImage);

	//��ͼ����з���˹��Ĥ���㴦��
	IGMImage(cropped_image,temp_1,IGM,radius,Ttest,thres_1);//�Ż��˺��������Ч��

	//ֱ��ͼ���⻯
	Histeq(temp_1,temp_2);

	//otsu��ֵ����ֵ��ͼ��
	cvThresholdOtsu(temp_2,temp_3);

	KillDot(temp_3,KDImage,(double)(thres_2*radius),features);

	//�жϱ�־
	int num_of_no_zero = cvCountNonZero(KDImage);
	
	//��ʾ�㷨�����еĽ��ͼ
	//	cvNamedWindow( "Original Image", 0 );
	//	cvNamedWindow( "Dealed Image1", 0 );
	//	cvNamedWindow( "Dealed Image2", 0 );
	//	cvNamedWindow( "Binary Image", 0 );
	//	cvNamedWindow( "KD Image", 0 );
	//	cvShowImage("Original Image",cropped_image);
	//	cvShowImage("Dealed Image1",temp_1);
	//	cvShowImage("Dealed Image2",temp_2);
	//	cvShowImage("Binary Image",temp_3);
	//	cvShowImage("KD Image",KDImage);
	//	cvWaitKey();
	//	cvDestroyAllWindows();

	//����������д���ı��ļ�
	Write_file(features,FEATURE_NUM,class_sign);

	//�ͷ��ڴ�
	delete [] IGM;
	IGM = NULL;	
	cvReleaseImage(&input);
	cvReleaseImage(&dst_1);
	cvReleaseImage(&dst_2);
	cvReleaseImage(&temp_1);
	cvReleaseImage(&temp_2);
	cvReleaseImage(&temp_3);
	cvReleaseImage(&KDImage);
	cvReleaseImage(&cropped_image);

	//�����жϽ��
	if (num_of_no_zero == 0)
		return 0;
	else
		return 1;

}

int CrackDetection::IsCrackTongue(const IplImage* src,int class_sign,int index,const double radius_ratio_thres,double thres_1,double thres_2)
{
	const int FEATURE_NUM = 9;
	double features[FEATURE_NUM];

	//��������ͼ�񸱱�
	IplImage* input = cvCreateImage(cvSize(src->width,src->height),src->depth,src->nChannels);
	cvCopy(src,input);

	//ȥ�������ɫ����
	IplImage* dst_1 = DeleteBlack(input);
	IplImage* dst_2 = ImageResize(dst_1,300);

	//����ȡ�����ֵĿ�Ⱥ͸߶�
	//int width = (int)((double)(src->width)*Ratio + 0.5);
	//int height = (int)((double)(src->height)*Ratio + 0.5);

	//�ü�ͼ����ȡ���о����󲿷�
	IplImage* cropped_image = ImageCrop(dst_2);

	//������Ĥ�뾶
	int radius = ((int)(cropped_image->height) > (int)(cropped_image->width))? ((int)(((double)(cropped_image->height))*radius_ratio_thres)):((int)(((double)(cropped_image->width))*radius_ratio_thres));

	//�뾶����
	features[0] = radius;

	//������Ĥ��������
	double* IGM = new double[(2*radius+1)*(2*radius+1)];

	//���㷴��˹��Ĥ����
	InverseGaussianMask(radius,IGM);

	//��ʾ����˹��Ĥ����
	//for(int i = 0;i<2*radius+1;i++)
	//{
	//	for (int j = 0;j<2*radius+1;j++)
	//	{
	//		cout<<IGM[i*(2*radius+1)+j]<<' ';
	//	}
	//	cout<<endl;
	//}

	//����tֵ
	double Ttest = Get_t(cropped_image);

	//���ڴ�ŷ���˹��Ĥ�������ͼ��
	IplImage* temp_1 = cvCreateImage(cvSize(cropped_image->width,cropped_image->height),cropped_image->depth,cropped_image->nChannels);
	cvSetZero(temp_1);

	//���ڴ��temp_1ֱ��ͼ���⻯���ͼ��
	IplImage* temp_2 = cvCreateImage(cvSize(cropped_image->width,cropped_image->height),cropped_image->depth,cropped_image->nChannels);
	cvSetZero(temp_2);

	//���ڴ��temp_2�Ķ�ֵ��ͼ��
	IplImage* temp_3 = cvCreateImage(cvSize(cropped_image->width,cropped_image->height),cropped_image->depth,cropped_image->nChannels);
	cvSetZero(temp_3);

	//���ڴ�����Ľ��ͼ
	IplImage* KDImage = cvCreateImage(cvSize(cropped_image->width,cropped_image->height),cropped_image->depth,cropped_image->nChannels);
	cvSetZero(KDImage);

	//��ͼ����з���˹��Ĥ���㴦��
	IGMImage(cropped_image,temp_1,IGM,radius,Ttest,thres_1);//�Ż��˺��������Ч��

	//ֱ��ͼ���⻯
	Histeq(temp_1,temp_2);

	//otsu��ֵ����ֵ��ͼ��
	cvThresholdOtsu(temp_2,temp_3);

	//double features[3];

	//ȥ���ӵ�
	KillDot(temp_3,KDImage,(double)(thres_2*radius),features);

	//�жϱ�־
	int num_of_no_zero = cvCountNonZero(KDImage);

	//��ʾ�㷨�����еĽ��ͼ
	//	cvNamedWindow( "Original Image", 0 );
	//	cvNamedWindow( "Dealed Image1", 0 );
	//	cvNamedWindow( "Dealed Image2", 0 );
	//	cvNamedWindow( "Binary Image", 0 );
	//	cvNamedWindow( "KD Image", 0 );
	//	cvShowImage("Original Image",cropped_image);
	//	cvShowImage("Dealed Image1",temp_1);
	//	cvShowImage("Dealed Image2",temp_2);
	//	cvShowImage("Binary Image",temp_3);
	//	cvShowImage("KD Image",KDImage);
	//	cvWaitKey();
	//	cvDestroyAllWindows();

	//����������д���ı��ļ�
	Write_file(features,FEATURE_NUM,class_sign,index);

	//�ͷ��ڴ�
	delete [] IGM;
	IGM = NULL;	
	cvReleaseImage(&input);
	cvReleaseImage(&dst_1);
	cvReleaseImage(&dst_2);
	cvReleaseImage(&temp_1);
	cvReleaseImage(&temp_2);
	cvReleaseImage(&temp_3);
	cvReleaseImage(&KDImage);
	cvReleaseImage(&cropped_image);

	//�����жϽ��
	if (num_of_no_zero == 0)
		return 0;
	else
		return 1;

}

int CrackDetection::IsCrackTongue(const IplImage* src,const char* dir,const double radius_ratio_thres,double thres_1,double thres_2)
{
	const int FEATURE_NUM = 8;
	double features[FEATURE_NUM];

	//��������ͼ�񸱱�
	IplImage* input = cvCreateImage(cvSize(src->width,src->height),src->depth,src->nChannels);
	cvCopy(src,input);

	//����ȡ�����ֵĿ�Ⱥ͸߶�
	//int width = (int)((double)(src->width)*Ratio + 0.5);
	//int height = (int)((double)(src->height)*Ratio + 0.5);

	//�ü�ͼ����ȡ���о����󲿷�
	IplImage* cropped_image = ImageCrop(input);

	//������Ĥ�뾶
	int radius = ((int)(cropped_image->height) > (int)(cropped_image->width))? ((int)(((double)(cropped_image->height))*radius_ratio_thres)):((int)(((double)(cropped_image->width))*radius_ratio_thres));

	//������Ĥ��������
	double* IGM = new double[(2*radius+1)*(2*radius+1)];

	//���㷴��˹��Ĥ����
	InverseGaussianMask(radius,IGM);

	//��ʾ����˹��Ĥ����
	//for(int i = 0;i<2*radius+1;i++)
	//{
	//	for (int j = 0;j<2*radius+1;j++)
	//	{
	//		cout<<IGM[i*(2*radius+1)+j]<<' ';
	//	}
	//	cout<<endl;
	//}

	//����tֵ
	double Ttest = Get_t(cropped_image);

	//���ڴ�ŷ���˹��Ĥ�������ͼ��
	IplImage* temp_1 = cvCreateImage(cvSize(cropped_image->width,cropped_image->height),cropped_image->depth,cropped_image->nChannels);
	cvSetZero(temp_1);

	//���ڴ��temp_1ֱ��ͼ���⻯���ͼ��
	IplImage* temp_2 = cvCreateImage(cvSize(cropped_image->width,cropped_image->height),cropped_image->depth,cropped_image->nChannels);
	cvSetZero(temp_2);

	//���ڴ��temp_2�Ķ�ֵ��ͼ��
	IplImage* temp_3 = cvCreateImage(cvSize(cropped_image->width,cropped_image->height),cropped_image->depth,cropped_image->nChannels);
	cvSetZero(temp_3);

	//���ڴ�����Ľ��ͼ
	IplImage* KDImage = cvCreateImage(cvSize(cropped_image->width,cropped_image->height),cropped_image->depth,cropped_image->nChannels);
	cvSetZero(KDImage);

	//��ͼ����з���˹��Ĥ���㴦��
	IGMImage(cropped_image,temp_1,IGM,radius,Ttest,thres_1);//�Ż��˺��������Ч��

	//ֱ��ͼ���⻯
	Histeq(temp_1,temp_2);

	//otsu��ֵ����ֵ��ͼ��
	cvThresholdOtsu(temp_2,temp_3);

	//double features[3];

	//ȥ���ӵ�
	KillDot(temp_3,KDImage,(double)(thres_2*radius),features);

	//�жϱ�־
	int num_of_no_zero = cvCountNonZero(KDImage);

	//��ʾ�㷨�����еĽ��ͼ
	//	cvNamedWindow( "Original Image", 0 );
	//	cvNamedWindow( "Dealed Image1", 0 );
	//	cvNamedWindow( "Dealed Image2", 0 );
	//	cvNamedWindow( "Binary Image", 0 );
	//	cvNamedWindow( "KD Image", 0 );
	//	cvShowImage("Original Image",cropped_image);
	//	cvShowImage("Dealed Image1",temp_1);
	//	cvShowImage("Dealed Image2",temp_2);
	//	cvShowImage("Binary Image",temp_3);
	//	cvShowImage("KD Image",KDImage);
	//	cvWaitKey();
	//	cvDestroyAllWindows();

	//����ͼ��
	cvSaveImage(dir,temp_3);

	//�ͷ��ڴ�
	delete [] IGM;
	IGM = NULL;	
	cvReleaseImage(&input);
	cvReleaseImage(&temp_1);
	cvReleaseImage(&temp_2);
	cvReleaseImage(&temp_3);
	cvReleaseImage(&KDImage);
	cvReleaseImage(&cropped_image);

	//�����жϽ��
	if (num_of_no_zero == 0)
		return 0;
	else
		return 1;

}

int CrackDetection::IsCrackTongue(const IplImage* src,const double radius_ratio_thres,double thres_1,double thres_2)
{
	//������ȡ����������
	const int FEATURE_NUM = 9;

	//���ڴ������ȡ������
	double features[FEATURE_NUM];

	//��������ͼ�񸱱�
	IplImage* input = cvCreateImage(cvSize(src->width,src->height),src->depth,src->nChannels);
	cvCopy(src,input);

	//ȥ�������ɫ����
	IplImage* dst_1 = DeleteBlack(input);
	IplImage* dst_2 = ImageResize(dst_1,300);

	//����ȡ�����ֵĿ�Ⱥ͸߶�
	//int width = (int)((double)(src->width)*Ratio + 0.5);
	//int height = (int)((double)(src->height)*Ratio + 0.5);

	//�ü�ͼ����ȡ���о����󲿷�
	IplImage* cropped_image = ImageCrop(dst_2);

	//������Ĥ�뾶
	int radius = ((int)(cropped_image->height) > (int)(cropped_image->width))? ((int)(((double)(cropped_image->height))*radius_ratio_thres)):((int)(((double)(cropped_image->width))*radius_ratio_thres));

	//�뾶����
	features[0] = radius;

	//������Ĥ��������
	double* IGM = new double[(2*radius+1)*(2*radius+1)];

	//���㷴��˹��Ĥ����
	InverseGaussianMask(radius,IGM);

	//����tֵ
	double Ttest = Get_t(cropped_image);

	//���ڴ�ŷ���˹��Ĥ�������ͼ��
	IplImage* temp_1 = cvCreateImage(cvSize(cropped_image->width,cropped_image->height),cropped_image->depth,cropped_image->nChannels);
	cvSetZero(temp_1);

	//���ڴ��temp_1ֱ��ͼ���⻯���ͼ��
	IplImage* temp_2 = cvCreateImage(cvSize(cropped_image->width,cropped_image->height),cropped_image->depth,cropped_image->nChannels);
	cvSetZero(temp_2);

	//���ڴ��temp_2�Ķ�ֵ��ͼ��
	IplImage* temp_3 = cvCreateImage(cvSize(cropped_image->width,cropped_image->height),cropped_image->depth,cropped_image->nChannels);
	cvSetZero(temp_3);

	//���ڴ�����Ľ��ͼ
	IplImage* KDImage = cvCreateImage(cvSize(cropped_image->width,cropped_image->height),cropped_image->depth,cropped_image->nChannels);
	cvSetZero(KDImage);

	//��ͼ����з���˹��Ĥ���㴦��
	IGMImage(cropped_image,temp_1,IGM,radius,Ttest,thres_1);//�Ż��˺��������Ч��

	//ֱ��ͼ���⻯
	Histeq(temp_1,temp_2);

	//otsu��ֵ����ֵ��ͼ��
	cvThresholdOtsu(temp_2,temp_3);

	//double features[3];

	//��ȡ����
	KillDot(temp_3,KDImage,(double)(thres_2*radius),features);

	//�жϱ�־
	//int num_of_no_zero = cvCountNonZero(KDImage);

	//�ͷ��ڴ�
	delete [] IGM;
	IGM = NULL;	
	cvReleaseImage(&input);
	cvReleaseImage(&dst_1);
	cvReleaseImage(&dst_2);
	cvReleaseImage(&temp_1);
	cvReleaseImage(&temp_2);
	cvReleaseImage(&temp_3);
	cvReleaseImage(&KDImage);
	cvReleaseImage(&cropped_image);

	//�����жϽ��
	//if (num_of_no_zero == 0)
		//return 0;
	//else
		//return 1;
	return 0;
}

void CrackDetection::GetFeatures(const IplImage* src,double features[9],const double radius_ratio_thres,double thres_1,double thres_2)
{
	//��������ͼ�񸱱�
	IplImage* input = cvCreateImage(cvSize(src->width,src->height),src->depth,src->nChannels);
	cvCopy(src,input);

	//ȥ�������ɫ����
	IplImage* dst_1 = DeleteBlack(input);
	IplImage* dst_2 = ImageResize(dst_1,300);

	//�ü�ͼ����ȡ���о����󲿷�
	IplImage* cropped_image = ImageCrop(dst_2);

	//showImage(dst_1, "DeleteBlack");
	//showImage(dst_2, "ImageResize");
	//showImage(cropped_image, "crop");

	//������Ĥ�뾶
	int radius = ((int)(cropped_image->height) > (int)(cropped_image->width))? ((int)(((double)(cropped_image->height))*radius_ratio_thres)):((int)(((double)(cropped_image->width))*radius_ratio_thres));

	//�뾶����
	features[0] = radius;

	//������Ĥ��������
	double* IGM = new double[(2*radius+1)*(2*radius+1)];

	//���㷴��˹��Ĥ����
	InverseGaussianMask(radius,IGM);

	//����tֵ
	double Ttest = Get_t(cropped_image);

	//���ڴ�ŷ���˹��Ĥ�������ͼ��
	IplImage* temp_1 = cvCreateImage(cvSize(cropped_image->width,cropped_image->height),cropped_image->depth,cropped_image->nChannels);
	cvSetZero(temp_1);

	//���ڴ��temp_1ֱ��ͼ���⻯���ͼ��
	IplImage* temp_2 = cvCreateImage(cvSize(cropped_image->width,cropped_image->height),cropped_image->depth,cropped_image->nChannels);
	cvSetZero(temp_2);

	//���ڴ��temp_2�Ķ�ֵ��ͼ��
	IplImage* temp_3 = cvCreateImage(cvSize(cropped_image->width,cropped_image->height),cropped_image->depth,cropped_image->nChannels);
	cvSetZero(temp_3);

	//���ڴ�����Ľ��ͼ
	IplImage* KDImage = cvCreateImage(cvSize(cropped_image->width,cropped_image->height),cropped_image->depth,cropped_image->nChannels);
	cvSetZero(KDImage);

	//��ͼ����з���˹��Ĥ���㴦��
	IGMImage(cropped_image,temp_1,IGM,radius,Ttest,thres_1);//�Ż��˺��������Ч��

	//ֱ��ͼ���⻯
	Histeq(temp_1,temp_2);

	//otsu��ֵ����ֵ��ͼ��
	cvThresholdOtsu(temp_2,temp_3);

	//showImage(temp_3, "tmp3");

	//��ȡ����
	KillDot(temp_3,KDImage,(double)(thres_2*radius),features);

	delete [] IGM;
	IGM = NULL;	
	cvReleaseImage(&input);
	cvReleaseImage(&dst_1);
	cvReleaseImage(&dst_2);
	cvReleaseImage(&temp_1);
	cvReleaseImage(&temp_2);
	cvReleaseImage(&temp_3);
	cvReleaseImage(&KDImage);
	cvReleaseImage(&cropped_image);
}