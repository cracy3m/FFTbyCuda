// FFTbyCuda.cpp : 定义控制台应用程序的入口点。
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
		waitKey(1);

		if ((src.type() == CV_16UC1) && getIsCudaAccLibOK()){
			//Mat floatmat = Mat_<float>(src);
			
			
			FFT_Complex *fftCdataPtr = nullptr;
			FFTPlan_Handle srcToFFTPlan = 0;

			if (src.data) {
				do {
					int res = 1;

					res = res && allocateFFTComplex(&fftCdataPtr,
						sizeof(FFT_Complex)*src.cols*src.rows
						);

					if (!res) break;

					res = res && createFFTPlan1d_C2C(&srcToFFTPlan, src.cols, src.rows);
					if (!res) break;

					//res = res && cudaMemFromHost(fftCdataPtr, floatmat.data,
					//	sizeof(float)*floatmat.cols*floatmat.rows);
					//if (!res) break;
					CuH_cpy16UC1ToDevComplex((unsigned short *)src.data, fftCdataPtr, src.cols, src.rows);

					res = res && execC2CfftPlan(srcToFFTPlan, fftCdataPtr, fftCdataPtr, 0);
					if (!res) break;

					
					
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

					CuH_logDevR2R(nullptr, src.cols , src.rows, 1.0f, nullptr);

					//magI = magI(Rect(0, 0, magI.cols / 2, magI.rows));	//.clone()

					//归一化处理，用0-1之间的浮点数将矩阵变换为可视的图像格式
					//normalize(magI, magI, 0, 1, CV_MINMAX);
					//Mat outMat = Mat_<unsigned short>(afterLog);
					//afterLog.convertTo(outMat, CV_16U, 4000.0);
					//imshow("outMat", outMat);
					//waitKey(1);

					Mat outMat;
					outMat.create(src.rows, src.cols, CV_16UC1);
					CuH_cvtDevRealTo16UC1(nullptr, src.cols, src.rows, 4000.0f, 0, (unsigned short*)outMat.data);

					imshow("outMat", outMat);
					waitKey(1);

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
		CuH_FreeTempCudaMem();
	}

    return 0;
}