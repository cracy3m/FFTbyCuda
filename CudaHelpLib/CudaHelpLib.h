#pragma once

#ifndef CUDA_HELP_LIB_H
#define CUDA_HELP_LIB_H

#ifdef _DLLEXPORTING
#define CUDAACCLIBDLL __declspec(dllexport)
#else
#define CUDAACCLIBDLL __declspec(dllimport)
#endif

#define CUDACALLMODE

typedef struct {
	char   name[256];                        /**< ASCII string identifying device */
	int    major;                            /**< Major compute capability */
	int    minor;                            /**< Minor compute capability */
	int    multiProcessorCount;              /**< Number of multiprocessors on device */
	int    deviceOverlap;                    /**< Device can concurrently copy memory and execute a kernel. Deprecated. Use instead asyncEngineCount. */
}_CUDA_DEV_INFO;

typedef struct {
	float re;
	float im;
} FFT_Complex;

typedef float FFT_Real;

typedef int FFTPlan_Handle;


#if defined(__cplusplus)||defined(c_plusplus)
extern "C" {
#endif

	CUDAACCLIBDLL void CUDACALLMODE setIsCudaAccLibOK(int isok);
	CUDAACCLIBDLL int CUDACALLMODE getIsCudaAccLibOK(void);

	CUDAACCLIBDLL _CUDA_DEV_INFO* CUDACALLMODE getCudaDeviceInfo(int id);
	CUDAACCLIBDLL int CUDACALLMODE getCudaDeviceCount(void);
	CUDAACCLIBDLL int CUDACALLMODE setCudaDevTo(int id);

	//create gpu temp memory
	CUDAACCLIBDLL void CUDACALLMODE cudaNop(void);
	//free gpu temp memory
	CUDAACCLIBDLL void CUDACALLMODE CuH_FreeTempCudaMem(void);

	CUDAACCLIBDLL int CUDACALLMODE cudaDevSync(void);


	//>>>>>>>>>>>>>>>>return 1 means exec ok//
	CUDAACCLIBDLL int CUDACALLMODE allocateFFTComplex(FFT_Complex** ptr, size_t size);
	CUDAACCLIBDLL int CUDACALLMODE allocateFFTComplexExt(FFT_Complex** ptr, int cols, int rows);

	CUDAACCLIBDLL int CUDACALLMODE allocateFFTReal(FFT_Real** ptr, size_t size);
	CUDAACCLIBDLL int CUDACALLMODE allocateFFTRealExt(FFT_Real** ptr, int cols, int rows);

	CUDAACCLIBDLL int CUDACALLMODE freeCudaMem(void *ptr);

	CUDAACCLIBDLL int CUDACALLMODE cudaMemFromHost(void *dstDev, void *srcHost, size_t byteSize);
	CUDAACCLIBDLL int CUDACALLMODE cudaMemToHost(void *dstHost, void *srcDev, size_t byteSize);
	CUDAACCLIBDLL int CUDACALLMODE cudaMemDevToDev(void *dstDev, void *srcDev, size_t byteSize);

	CUDAACCLIBDLL int CUDACALLMODE destroyFFTPlan(FFTPlan_Handle plan);
	CUDAACCLIBDLL int CUDACALLMODE createFFTPlan1d_R2C(FFTPlan_Handle *plan, int cols, int rows);
	CUDAACCLIBDLL int CUDACALLMODE createFFTPlan1d_C2C(FFTPlan_Handle *plan, int cols, int rows);
	CUDAACCLIBDLL int CUDACALLMODE createFFTPlan1d_C2R(FFTPlan_Handle *plan, int cols, int rows);

	CUDAACCLIBDLL int CUDACALLMODE execR2CfftPlan(FFTPlan_Handle plan, FFT_Real *idata, FFT_Complex *odata);

	//@param direction  0:FFT(forwart) 1:IFFT(inverse)
	CUDAACCLIBDLL int CUDACALLMODE execC2CfftPlan(FFTPlan_Handle plan, FFT_Complex *idata, FFT_Complex *odata,int direction);
	
	CUDAACCLIBDLL int CUDACALLMODE execC2RfftPlan(FFTPlan_Handle plan, FFT_Complex *idata, FFT_Real *odata);

	//<<<<<<<<<<<<<<return 1 means exec ok//

	//download gpu interal dev_temp_4M2 buffer to host_dst
	//return 0 means exec ok
	CUDAACCLIBDLL int CUDACALLMODE CuH_downloadTemp4M2(int size, unsigned char* host_dst);


	//upload host_src to gpu interal dev_temp_4M2 buffer
	//return 0 means exec ok
	CUDAACCLIBDLL int CUDACALLMODE CuH_uploadTemp4M2(int size, unsigned char* host_src);


	//calc magnitude from dev complex data to host real data
	//if devSrc==nullptr , use interal dev_temp_4M2 buffer as src data
	//if hostDst==nullptr , the calc result copy to interal dev_temp_4M2 buffer
	//return 0 means exec ok
	CUDAACCLIBDLL int CUDACALLMODE CuH_magnitudeDevC2R(FFT_Complex *devSrc, int cols, int rows, FFT_Real *hostDst);

	//calc log from dev real data to host real data
	//if devSrc==nullptr , use interal dev_temp_4M2 buffer as src data
	//if hostDst==nullptr , the calc result copy to interal dev_temp_4M2 buffer
	// @param beta is the value add to data
	//return 0 means exec ok
	CUDAACCLIBDLL int CUDACALLMODE CuH_logDevR2R(FFT_Real *devSrc, int cols, int rows, float beta, FFT_Real *hostDst);

	//convert dev real data to host 16UC1 data
	//if devSrc==nullptr , use interal dev_temp_4M2 buffer as src data
	//if hostDst==nullptr , the calc result copy to interal dev_temp_4M2 buffer
	// @param alpha is the value scale the data
	// @param beta is the value add to scaled data
	//return 0 means exec ok
	CUDAACCLIBDLL int CUDACALLMODE CuH_cvtDevRealTo16UC1(FFT_Real *devSrc, int cols, int rows, float alpha, float beta, unsigned short *hostDst);

	//copy host Real data to cuda Complex data 
	//set dstDev[].Re=srcHost[] dstDev[].Im=0
	//if devSrc==nullptr , use interal dev_temp_4M2 buffer as src data
	//return 0 means exec ok
	CUDAACCLIBDLL int CUDACALLMODE CuH_cpyHostRealToDevComplex(FFT_Real *srcHost, FFT_Complex *dstDev, int cols, int rows);

	//copy host 16UC1 data to cuda Complex data 
	//set dstDev[].Re=srcHost[] dstDev[].Im=0
	//if devSrc==nullptr , use interal dev_temp_4M2 buffer as src data
	//return 0 means exec ok
	CUDAACCLIBDLL int CUDACALLMODE CuH_cpy16UC1ToDevComplex(unsigned short *srcHost, FFT_Complex *dstDev, int cols, int rows);

	//ROI of data[rows][cols] to  data[height][width] of RECT{(x,y),(width,height)}
	//return 0 means exec ok
	CUDAACCLIBDLL int CUDACALLMODE CuH_ROIdevComplex(FFT_Complex *dataDev, int cols, int rows, int x, int y, int width, int height);

	//transpose FFT_Complex cuda image data
	//@param rows is before transpose size
	//@param cols is before transpose size
	//return 0 means exec ok
	CUDAACCLIBDLL int CUDACALLMODE CuH_transposeComplex(int rows, int cols, FFT_Complex* dev_src, FFT_Complex *dev_dst);

	//transpose cuda float image data
	//if dev_src==nullptr, use interal dev_temp_4M2 buffer as source data
	//if dev_dst==nullptr, the calc result copy to interal dev_temp_4M2 buffer
	//@param rows is before transpose size
	//@param cols is before transpose size
	//return 0 means exec ok
	CUDAACCLIBDLL int CUDACALLMODE CuH_transpose32FC1(int rows, int cols, void* dev_src, void *dev_dst);

	//transpose interal dev_temp_4M2
	//if host_src==nullptr, use interal dev_temp_4M2 buffer as source data
	//@param rows is before transpose size
	//@param cols is before transpose size
	//return 0 means exec ok
	CUDAACCLIBDLL int CUDACALLMODE CuH_transpose16UC1(int rows, int cols, void* host_src, void *host_dst);

	//dataDev calc with fft window data winDev and dispersion data dispersionDev
	//return 0 means exec ok
	CUDAACCLIBDLL int CUDACALLMODE CuH_devCdataCalcWinAndDispersion(int cols, int rows, FFT_Complex *dataDev, FFT_Real *winDev, FFT_Complex *dispersionDev);

	//use interal dev_temp_4M2 buffer to processe power(x,p),then store the result to dev_temp_4M2
	//return 0 means exec ok
	CUDAACCLIBDLL int CUDACALLMODE CuH_power8UC1(int rows, int cols, float p);

	// convert 16UC1 data (host_src) to 8UC1 data, { resualt=(255/winWidth)*(data-winCenter+winWidth/2) }
	// if host_src == nullptr, use interal dev_temp_4M2 buffer as source data
	//return 0 means exec ok
	CUDAACCLIBDLL int CUDACALLMODE CuH_pixWindow16UC1To8UC1(int rows, int cols, int winCenter, int winWidth, unsigned short *host_src);

	//calc all image pixels average value
	//if host_src==nullptr, use interal dev_temp_4M2 buffer as source data
	//return 0 means exec ok
	CUDAACCLIBDLL int CUDACALLMODE CuH_allPixAvgValue(int rows, int cols, unsigned short* host_src, float *host_res);

	//threshold 16UC1 data 
	//if host_src==nullptr, use interal dev_temp_4M2 buffer as source data
	//@param mode : bit_0=1 mean normal threshold {data=(data>=threshold)*data} else {data=(data>=threshold)*65535}
	//				bit_7=1 mean normal threshold {data=(data<threshold)*data} else {data=(data<threshold)*65535}
	//return 0 means exec ok
	CUDAACCLIBDLL int CUDACALLMODE CuH_threshold16UC1(int rows, int cols, int thres, int mode, unsigned short* host_src);

	//set dataDev left side to zero,and then divided by N
	//return 0 means exec ok
	CUDAACCLIBDLL int CUDACALLMODE CuH_zeroLeftComplexAndDivConst(int rows, int cols, float divConst, FFT_Complex *dataDev);

	//set dataDev Re or Im to zero 
	//@param mode  0:zero Im , 1:zero Re
	//return 0 means exec ok
	CUDAACCLIBDLL int CUDACALLMODE CuH_zeroComplexReOrIm(int rows, int cols, int mode, FFT_Complex *dataDev);

	// use interal dev_temp_4M2 buffer as source data , do Horizontal flip
	// return 0 means exec ok
	CUDAACCLIBDLL int CUDACALLMODE CuH_flipH8UC1(int rows, int cols);

#if defined(__cplusplus)||defined(c_plusplus)
}
#endif

#endif	//CUDA_HELP_LIB_H