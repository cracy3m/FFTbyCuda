// FFTbyCuda.cpp : �������̨Ӧ�ó������ڵ㡣
//

#include "stdafx.h"
#include <stdlib.h>
#include <iostream>
#include <math.h>
#include "string.h"

#include "opencv.hpp"
#include "CudaHelpLib.h"

using namespace std;
using namespace cv;

//�����������������
//void  GenerateWindowFuction(ScanParam *psPtn) {
//	unsigned int wPixelNum = psPtn->wPixelNum;
//	unsigned int sWindowType = psPtn->nWindowType;
//	unsigned int N_1 = wPixelNum - 1;
//
//	double dt = 0;
//
//	//calculate the window data
//	switch (sWindowType)
//	{
//	case Hanning:
//		for (int i = 0; i < wPixelNum; i++)
//		{
//			//	dt = cos((2.0*PI*i) / N_1);
//			//	dt = 0.5 - 0.5*dt;
//			//	m_pWinFuc[i] = dt;
//			dt = (2.0*i / N_1) - 1;
//			m_pWinFuc[i] = 0.5 + 0.5*cos(PI*(dt - 4 / wPixelNum));
//
//		}
//		break;
//	case Gaussian:
//		for (int i = 0; i < wPixelNum; i++)
//		{
//			dt = 6.0 * i / N_1;
//			dt = -0.5*dt*dt;
//			dt = exp(dt);
//			m_pWinFuc[i] = dt;
//		}
//		break;
//	case Blackman:
//		do {
//			double dt1 = 0;
//			for (int i = 0; i < wPixelNum; i++)
//			{
//				dt = 0.008*cos(4.0*PI*i / N_1);
//				dt1 = 0.5*cos(2 * PI*i / N_1);
//				m_pWinFuc[i] = 0.42 - dt1 + dt;
//			}
//		} while (0);
//		break;
//
//	default:
//		for (int i = 0; i < MAX_SPOT_NUM; i++)
//		{
//			m_pWinFuc[i] = 1;
//		}
//		break;
//	}
//
//
//}

//����������������� data[N]
void  GenerateWinFuncToReal(int N, FFT_Real *data) {
	if (!data) return;
	unsigned int N_1 = N - 1;
	double dt;
	for (int i = 0; i < N; i++)
	{
		//dt = cos((2.0*CV_PI*i) / N_1);
		//dt = 0.5 - 0.5*dt;
		//data[i] = dt;

		dt = (2.0*i / N_1) - 1;
		data[i] = 0.5 + 0.5*cos(CV_PI*(dt - 4.0 / N));
	}
}

//���� ɫɢ���� ��һά�������� data[N]
void LoadDispersionToComplex(int N, double a2, double a3, FFT_Complex *data) {
	if (!data) return;
	double dv;
	for (int i = 0; i < N; i++)
	{
		dv = -i*i*(a3*i + a2);
		data[i].re = cos(dv);
		data[i].im = sin(dv);
	}
}

void initCudaAccLib() {
	if (getIsCudaAccLibOK()) return;

	int devcnt;
	int gpu_device_id = -1;
	bool m_isCudaAccLibOK = false;
	devcnt = getCudaDeviceCount();

	if (devcnt<1) {
		cout << "opencv has no cuda support.";
		m_isCudaAccLibOK = false;
	}
	else {
		_CUDA_DEV_INFO *pCudaInfo;
		for (int i = 0; i<devcnt; i++) {
			pCudaInfo = getCudaDeviceInfo(i);
			if (pCudaInfo->deviceOverlap != 0) {
				gpu_device_id = i;
				break;
			}
		}

		if (gpu_device_id != -1) {
			if (!setCudaDevTo(gpu_device_id)) gpu_device_id = -1;
		}

		if (gpu_device_id != -1) {
			char str[256] = { 0 };
			sprintf(str, "GPU name= %s\n", pCudaInfo->name);
			cout << str;

			sprintf(str, "version=%d .%d\n", pCudaInfo->major, pCudaInfo->minor);
			cout << str;

			sprintf(str, "multiProcessorCount = %d\n", pCudaInfo->multiProcessorCount);
			cout << str;

			sprintf(str, "current cuda dev id = %d\n", gpu_device_id);
			cout << str;

			cudaNop();

			m_isCudaAccLibOK = true;
		}
		else {
			m_isCudaAccLibOK = false;
		}
	}

	setIsCudaAccLibOK(m_isCudaAccLibOK);
}

void transposeAndFFTbyCV(Mat &img) {
	Mat ttt;
	cv::transpose(img, ttt);
	imshow("ttt", ttt);
	cv::waitKey(1);

	Mat padded;                 //��0�������ͼ�����
	int m = ttt.rows; //getOptimalDFTSize(disp.rows);
	int n = getOptimalDFTSize(ttt.cols);

	//�������ͼ��I���������Ϊpadded���Ϸ����󷽲�����䴦��
	copyMakeBorder(ttt, padded, 0, m - ttt.rows, 0, n - ttt.cols, BORDER_CONSTANT, Scalar::all(0));
	Mat planes[] = { Mat_<float>(padded), Mat::zeros(padded.size(),CV_32F) };
	Mat complexI;
	merge(planes, 2, complexI);     //��planes�ںϺϲ���һ����ͨ������complexI
	dft(complexI, complexI, DFT_ROWS);        //���и���Ҷ�任

	split(complexI, planes);        //planes[0] = Re(DFT(I),planes[1] = Im(DFT(I))
									//��planes[0]Ϊʵ��,planes[1]Ϊ�鲿
	magnitude(planes[0], planes[1], planes[0]);     //planes[0] = magnitude
	Mat magI = planes[0];
	magI += Scalar::all(1);
	log(magI, magI);                //ת���������߶�(logarithmic scale)

	magI = magI(Rect(0, 0, magI.cols / 2, magI.rows));	//.clone()

														//��һ��������0-1֮��ĸ�����������任Ϊ���ӵ�ͼ���ʽ
														//normalize(magI, magI, 0, 1, CV_MINMAX);
	Mat outMat = Mat_<unsigned short>(magI);
	magI.convertTo(outMat, CV_16U, 4000.0);
	imshow("ttt FFT Output Mat", outMat);

	waitKey(0);
}

int main()
{
	Mat src = imread("./../../ats_raw_wave.png", CV_LOAD_IMAGE_UNCHANGED);

	if (!src.empty() && src.data) {
		initCudaAccLib();
		imshow("src", src);
		cv::waitKey(1);

		if ((src.type() == CV_16UC1) && getIsCudaAccLibOK()){
			//Mat floatmat = Mat_<float>(src);
			
			
			FFT_Complex *fftCdataPtr = nullptr;
			FFTPlan_Handle srcToFFTPlan = 0;

			FFT_Complex *fftCtranPtr = nullptr;
			FFTPlan_Handle tranFFTPlan = 0;
			

			FFT_Real *fftWinFuncPtr = nullptr;
			FFT_Complex *dispersionCPtr = nullptr;
			FFT_Real *hostWinFunc = nullptr;
			FFT_Complex *hostDispersion = nullptr;

			if (src.data) {

				//transposeAndFFTbyCV(src);
				
				do {
					int res = 1;

					res = res && allocateFFTComplex(&fftCtranPtr,
						sizeof(FFT_Complex)*src.cols*src.rows
						);

					if (!res) break;

					res = res && allocateFFTComplex(&fftCdataPtr,
						sizeof(FFT_Complex)*src.cols*src.rows
						);

					if (!res) break;

					res = res && allocateFFTReal(&fftWinFuncPtr,
						sizeof(FFT_Real)*src.cols
						);

					if (!res) break;

					res = res && allocateFFTComplex(&dispersionCPtr,
						sizeof(FFT_Complex)*src.cols
						);

					if (!res) break;

					res = res && createFFTPlan1d_C2C(&srcToFFTPlan, src.cols, src.rows);
					if (!res) break;

					res = res && createFFTPlan1d_C2C(&tranFFTPlan, src.rows, src.cols);
					if (!res) break;


					hostWinFunc = new FFT_Real[src.cols];
					hostDispersion = new FFT_Complex[src.cols];
					GenerateWinFuncToReal(src.cols, hostWinFunc);
					LoadDispersionToComplex(src.cols, 0, 0, hostDispersion);

					cudaMemFromHost(fftWinFuncPtr, hostWinFunc, sizeof(FFT_Real)*src.cols);
					cudaMemFromHost(dispersionCPtr, hostDispersion, sizeof(FFT_Complex)*src.cols);

					//res = res && cudaMemFromHost(fftCdataPtr, floatmat.data,
					//	sizeof(float)*floatmat.cols*floatmat.rows);
					//if (!res) break;
					CuH_cpy16UC1ToDevComplex((unsigned short *)src.data, fftCdataPtr, src.cols, src.rows);

					//������ ԭʼ�������� ת��  ������ fft
					Mat tran_src(src.cols, src.rows,CV_16UC1);

					CuH_transposeComplex(src.rows, src.cols, fftCdataPtr, fftCtranPtr);
					
					res = res && execC2CfftPlan(tranFFTPlan, fftCtranPtr, fftCtranPtr, 0);
					if (!res) break;
/*
					CuH_ROIdevComplex(fftCtranPtr, tran_src.cols, tran_src.rows, 0, 0, tran_src.cols/2  , tran_src.rows);

					tran_src.create(tran_src.rows, tran_src.cols / 2, CV_16UC1);

					CuH_magnitudeDevC2R(fftCtranPtr, tran_src.cols, tran_src.rows, nullptr);

					CuH_logDevR2R(nullptr, tran_src.cols, tran_src.rows, 1.0f, nullptr);

					CuH_cvtDevRealTo16UC1(nullptr, tran_src.cols , tran_src.rows, 4000.0f, 0, (unsigned short*)tran_src.data);

					imshow("tran_fft", tran_src);
					cv::waitKey(1);
*/
					Mat cttt;
					cttt.create(tran_src.rows, tran_src.cols, CV_32FC2);
					if (cttt.data) {	//cttt.data
						cudaMemToHost(cttt.data, fftCtranPtr, tran_src.rows*tran_src.cols*sizeof(FFT_Complex));
						FFT_Complex *ptr = (FFT_Complex *)cttt.data;
						
						for (int j = 0; j < tran_src.rows; j++)
						{
							for (int i = tran_src.cols / 2; i < tran_src.cols ; i++) //tran_src.cols/2
							{
								ptr[j*tran_src.cols + i].re /= tran_src.cols;
								ptr[j*tran_src.cols + i].im /= tran_src.cols;
							}
						}

						for (int j = 0; j < tran_src.rows; j++)
						{
							for (int i = 0; i < tran_src.cols/2; i++) //tran_src.cols/2
							{
								ptr[j*tran_src.cols + i].re = 0;
								ptr[j*tran_src.cols + i].im = 0;
							}
						}
						cudaMemFromHost(fftCtranPtr, cttt.data, tran_src.rows*tran_src.cols*sizeof(FFT_Complex));
						if (execC2CfftPlan(tranFFTPlan, fftCtranPtr, fftCtranPtr, 1)) {
							CuH_transposeComplex(tran_src.rows, tran_src.cols, fftCtranPtr,fftCdataPtr );
						}
					}

					// ��fft�� �� ����ɫɢ ��������
					CuH_devCdataCalcWinAndDispersion(src.cols, src.rows, fftCdataPtr, fftWinFuncPtr, dispersionCPtr);

					res = res && execC2CfftPlan(srcToFFTPlan, fftCdataPtr, fftCdataPtr, 0);
					if (!res) break;

					CuH_ROIdevComplex(fftCdataPtr, src.cols, src.rows, 0, 0, src.cols / 2, src.rows);
					
					CuH_magnitudeDevC2R(fftCdataPtr, src.cols / 2, src.rows, nullptr);

					//cudaMemToHost(after1stFFTBufptr, fftCdataPtr, sizeof(FFT_Complex)*(floatmat.cols / 2 + 1)*floatmat.rows);
					//Mat planes[2];
					//split(after1stFFTCMat, planes);
					//magnitude(planes[0], planes[1], planes[0]);     //planes[0] = magnitude
					//Mat magI = planes[0];
					//magI += Scalar::all(1);
					//log(magI, magI);                //ת���������߶�(logarithmic scale)
					//afterMagnitude += Scalar::all(1);
					//log(afterMagnitude, afterMagnitude);

					CuH_logDevR2R(nullptr, src.cols/2 , src.rows, 1.0f, nullptr);

					//magI = magI(Rect(0, 0, magI.cols / 2, magI.rows));	//.clone()

					//��һ��������0-1֮��ĸ�����������任Ϊ���ӵ�ͼ���ʽ
					//normalize(magI, magI, 0, 1, CV_MINMAX);
					//Mat outMat = Mat_<unsigned short>(afterLog);
					//afterLog.convertTo(outMat, CV_16U, 4000.0);
					//imshow("outMat", outMat);
					//waitKey(1);

					
					CuH_cvtDevRealTo16UC1(nullptr, src.cols/2, src.rows, 2000.0f, 0, nullptr);

					Mat outMat;
					outMat.create(src.cols / 2, src.rows, CV_8UC1);

					CuH_transpose16UC1(outMat.cols, outMat.rows, nullptr, nullptr);

					float avg = 0.0f;
					CuH_allPixAvgValue(outMat.rows, outMat.cols, nullptr, &avg);

					avg *= 0;//1.15;
					CuH_threshold16UC1(outMat.rows, outMat.cols, (unsigned short)avg, 1, nullptr);

					CuH_pixWindow16UC1To8UC1(outMat.rows, outMat.cols, 32767, 30000, nullptr);

					CuH_power8UC1(outMat.rows, outMat.cols, 1.3f);

					CuH_downloadTemp4M2(outMat.rows* outMat.cols, outMat.data);

					for (int i = 0; i < 8; i++)
					{
						for (int j = 0; j < outMat.cols; j++)
						{
							outMat.data[i*outMat.cols + j] = 0;
						}
					}

					

					resize(outMat, outMat, Size(512, 1024));

					imwrite("./../../result.png", outMat);

					imshow("outMat", outMat);
					cv::waitKey(1);

				} while (0);
			}

			if (srcToFFTPlan) {
				destroyFFTPlan(srcToFFTPlan);
			}

			if (tranFFTPlan) {
				destroyFFTPlan(tranFFTPlan);
			}

			if (fftWinFuncPtr) {
				freeCudaMem(fftWinFuncPtr);
			}

			if (dispersionCPtr) {
				freeCudaMem(dispersionCPtr);
			}

			if (hostWinFunc) {
				delete[] hostWinFunc;
			}

			if (hostDispersion) {
				delete[] hostDispersion;
			}

			if (fftCtranPtr) {
				freeCudaMem(fftCtranPtr);
			}

			if (fftCdataPtr) {
				freeCudaMem(fftCdataPtr);
			}

		}

		cv::waitKey(0);
		CuH_FreeTempCudaMem();
	}

    return 0;
}