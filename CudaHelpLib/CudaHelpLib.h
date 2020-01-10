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
	//if data==nullptr , use interal dev_temp_4M2 buffer as precess data
	//return 0 means exec ok
	CUDAACCLIBDLL int CUDACALLMODE CuH_ROIdevComplex(FFT_Complex *dataDev, int cols, int rows, int x, int y, int width, int height);

	//transpose interal dev_temp_4M2
	//if host_src==nullptr, use interal dev_temp_4M2 buffer as source data
	//@param rows is before transpose size
	//@param cols is before transpose size
	//return 0 means exec ok
	CUDAACCLIBDLL int CUDACALLMODE CuH_transpose16UC1(int rows, int cols, void* host_src, void *host_dst);

#if defined(__cplusplus)||defined(c_plusplus)
}
#endif

#endif	//CUDA_HELP_LIB_H