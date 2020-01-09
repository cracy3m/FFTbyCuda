// FFTbyCuda.cpp : �������̨Ӧ�ó������ڵ㡣
//

#include "stdafx.h"
#include <stdlib.h>
#include <iostream>
#include "string.h"

#include "opencv.hpp"
#include "CudaHelpLib.h"

using namespace std;
using namespace cv;

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
		
		if ((src.type() == CV_16UC1) && getIsCudaAccLibOK()){
			Mat floatmat = Mat_<float>(src);
			
			
			FFT_Complex *fftCdataPtr = nullptr;
			FFTPlan_Handle srcToFFTPlan = 0;

			if (floatmat.data) {
				do {
					int res = 1;

					res = res && allocateFFTComplex(&fftCdataPtr,
						sizeof(FFT_Complex)*(floatmat.cols / 2 + 1)*floatmat.rows
						);

					if (!res) break;

					res = res && createFFTPlan1d_R2C(&srcToFFTPlan, floatmat.cols, floatmat.rows);
					if (!res) break;

					res = res && cudaMemFromHost(fftCdataPtr, floatmat.data,
						sizeof(float)*floatmat.cols*floatmat.rows);
					if (!res) break;

					res = res && execR2CfftPlan(srcToFFTPlan, (FFT_Real*)fftCdataPtr, fftCdataPtr);
					if (!res) break;

					float *after1stFFTBufptr = new float[sizeof(FFT_Complex)*(floatmat.cols / 2 + 1)*floatmat.rows / sizeof(float)];
					Mat after1stFFTCMat(floatmat.rows, floatmat.cols/2 + 1, CV_32FC2, after1stFFTBufptr);

					cudaMemToHost(after1stFFTBufptr, fftCdataPtr, sizeof(FFT_Complex)*(floatmat.cols / 2 + 1)*floatmat.rows);

					Mat planes[2];
					split(after1stFFTCMat, planes);
					magnitude(planes[0], planes[1], planes[0]);     //planes[0] = magnitude
					Mat magI = planes[0];
					magI += Scalar::all(1);
					log(magI, magI);                //ת���������߶�(logarithmic scale)

					//magI = magI(Rect(0, 0, magI.cols / 2, magI.rows));	//.clone()

					//��һ����������0-1֮��ĸ�����������任Ϊ���ӵ�ͼ���ʽ
					//normalize(magI, magI, 0, 1, CV_MINMAX);
					Mat outMat = Mat_<unsigned short>(magI);
					magI.convertTo(outMat, CV_16U, 4000.0);
					imshow("outMat", outMat);

					delete[] after1stFFTBufptr;

				} while (0);
			}

			if (fftCdataPtr) {
				freeCudaMem(fftCdataPtr);
			}


			if (srcToFFTPlan) {
				destroyFFTPlan(srcToFFTPlan);
			}
		}

		waitKey(0);
	}

    return 0;
}
