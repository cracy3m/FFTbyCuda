// FFTbyCuda.cpp : 定义控制台应用程序的入口点。
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

//计算产生窗函数数据
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

//计算产生窗函数数据 data[N]
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

//计算 色散数据 到一维复数数组 data[N]
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

			FFT_Real *fftWinFuncPtr = nullptr;
			FFT_Complex *dispersionCPtr = nullptr;
			FFT_Real *hostWinFunc = nullptr;
			FFT_Complex *hostDispersion = nullptr;

			if (src.data) {
				do {
					int res = 1;

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

					// 开fft窗 和 乘以色散 复数数组
					CuH_devCdataCalcWinAndDispersion(src.cols, src.rows, fftCdataPtr, fftWinFuncPtr, dispersionCPtr);

					res = res && execC2CfftPlan(srcToFFTPlan, fftCdataPtr, fftCdataPtr, 0);
					if (!res) break;

					CuH_ROIdevComplex(fftCdataPtr, src.cols, src.rows, 0, 0, src.cols / 2, src.rows);
					
					CuH_magnitudeDevC2R(fftCdataPtr, src.cols , src.rows, nullptr);
					

					//cudaMemToHost(after1stFFTBufptr, fftCdataPtr, sizeof(FFT_Complex)*(floatmat.cols / 2 + 1)*floatmat.rows);
					//Mat planes[2];
					//split(after1stFFTCMat, planes);
					//magnitude(planes[0], planes[1], planes[0]);     //planes[0] = magnitude
					//Mat magI = planes[0];
					//magI += Scalar::all(1);
					//log(magI, magI);                //转换到对数尺度(logarithmic scale)
					//afterMagnitude += Scalar::all(1);
					//log(afterMagnitude, afterMagnitude);

					CuH_logDevR2R(nullptr, src.cols/2 , src.rows, 1.0f, nullptr);

					//magI = magI(Rect(0, 0, magI.cols / 2, magI.rows));	//.clone()

					//归一化处理，用0-1之间的浮点数将矩阵变换为可视的图像格式
					//normalize(magI, magI, 0, 1, CV_MINMAX);
					//Mat outMat = Mat_<unsigned short>(afterLog);
					//afterLog.convertTo(outMat, CV_16U, 4000.0);
					//imshow("outMat", outMat);
					//waitKey(1);

					
					CuH_cvtDevRealTo16UC1(nullptr, src.cols/2, src.rows, 4000.0f, 0, nullptr);

					Mat outMat;
					outMat.create(src.cols / 2, src.rows, CV_16UC1);
					CuH_transpose16UC1(outMat.cols, outMat.rows, nullptr, outMat.data);

					resize(outMat, outMat, Size(512, 800));

					imshow("outMat", outMat);
					cv::waitKey(1);

				} while (0);
			}

			if (srcToFFTPlan) {
				destroyFFTPlan(srcToFFTPlan);
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

			if (fftCdataPtr) {
				freeCudaMem(fftCdataPtr);
			}

		}

		cv::waitKey(0);
		CuH_FreeTempCudaMem();
	}

    return 0;
}