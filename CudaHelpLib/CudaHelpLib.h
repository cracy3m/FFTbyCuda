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

	CUDAACCLIBDLL int CUDACALLMODE cudaDevSync(void);

	CUDAACCLIBDLL int CUDACALLMODE allocateFFTComplex(FFT_Complex** ptr, size_t size);
	CUDAACCLIBDLL int CUDACALLMODE allocateFFTComplexExt(FFT_Complex** ptr, int cols, int rows);

	CUDAACCLIBDLL int CUDACALLMODE allocateFFTReal(FFT_Real** ptr, size_t size);
	CUDAACCLIBDLL int CUDACALLMODE allocateFFTRealExt(FFT_Real** ptr, int cols, int rows);

	CUDAACCLIBDLL int CUDACALLMODE freeCudaMem(void *ptr);

	CUDAACCLIBDLL int CUDACALLMODE cudaMemFromHost(void *dstDev, void *srcHost, size_t byteSize);
	CUDAACCLIBDLL int CUDACALLMODE cudaMemToHost(void *dstHost, void *srcDev, size_t byteSize);
	CUDAACCLIBDLL int CUDACALLMODE cudaMemDevToDev(void *dstDev, void *srcDev, size_t byteSize);

	CUDAACCLIBDLL int CUDACALLMODE createFFTPlan1d_R2C(FFTPlan_Handle *plan, int cols, int rows);
	CUDAACCLIBDLL int CUDACALLMODE destroyFFTPlan(FFTPlan_Handle plan);

	CUDAACCLIBDLL int CUDACALLMODE execR2CfftPlan(FFTPlan_Handle plan, FFT_Real *idata, FFT_Complex *odata);

	

#if defined(__cplusplus)||defined(c_plusplus)
}
#endif

#endif	//CUDA_HELP_LIB_H