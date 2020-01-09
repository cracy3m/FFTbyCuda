
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cufft.h"

#include <stdio.h>

#include "CudaHelpLib.h"

static int isCudaOk = 0;
static _CUDA_DEV_INFO deviceinfo;

void CUDACALLMODE setIsCudaAccLibOK(int isok) {
	isCudaOk = isok;
}

int CUDACALLMODE getIsCudaAccLibOK(void) {
	return isCudaOk;
}

_CUDA_DEV_INFO* CUDACALLMODE getCudaDeviceInfo(int id) {
	cudaFree(0);
	if (id >= 0 || id <= 4)
	{
		cudaError_t res;
		cudaDeviceProp device_prop;
		res = cudaSetDevice(id);

		if (res != cudaSuccess) {
			fprintf(stderr, "invaild cuda id!");
			return &deviceinfo;
		}
		device_prop.name[0] = 0;
		cudaGetDeviceProperties(&device_prop, id);
		sprintf(deviceinfo.name, "%s", device_prop.name);
		deviceinfo.major = device_prop.major;
		deviceinfo.minor = device_prop.minor;
		deviceinfo.multiProcessorCount = device_prop.multiProcessorCount;
		deviceinfo.deviceOverlap = device_prop.deviceOverlap;
	}
	else {
		deviceinfo.name[0] = 0;
	}
	
	return &deviceinfo;
}

int getCudaDeviceCount(void) {
	int device_count;
	cudaGetDeviceCount(&device_count);
	return device_count;
}

int  setCudaDevTo(int id) {
	cudaError_t cudaStatus = cudaSuccess;
	cudaStatus = cudaSetDevice(id);
	if (cudaStatus == cudaSuccess) {
		return 1;
	}
	else {
		return 0;
	}
}

int cudaDevSync(void) {
	cudaError_t cudaStatus = cudaSuccess;
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus == cudaSuccess) {
		return 1;
	}
	else {
		return 0;
	}
}

int allocateFFTComplex(FFT_Complex** ptr, size_t size) {
	if (size < 1) {
		*ptr = nullptr;
		return 0;
	}

	cudaError_t cudaStatus = cudaSuccess;
	FFT_Complex *res = nullptr;
	cudaStatus = cudaMalloc((void **)&res, size);
	if (cudaStatus == cudaSuccess && res) {
		*ptr = res;
		return 1;
	}
	else {
		*ptr = nullptr;
		return 0;
	}
}

int allocateFFTComplexExt(FFT_Complex** ptr, int cols, int rows) {
	
	if (cols < 0) {
		*ptr = nullptr;
		return 0;
	}

	if (rows < 0) rows = 1;

	size_t size = static_cast<size_t>(cols)*static_cast<size_t>(rows)*sizeof(FFT_Complex);

	cudaError_t cudaStatus = cudaSuccess;
	FFT_Complex *res = nullptr;
	cudaStatus = cudaMalloc((void **)&res, size);
	if (cudaStatus == cudaSuccess && res) {
		*ptr = res;
		return 1;
	}
	else {
		*ptr = nullptr;
		return 0;
	}

}

int allocateFFTReal(FFT_Real** ptr, size_t size) {
	if (size < 1) {
		*ptr = nullptr;
		return 0;
	}

	cudaError_t cudaStatus = cudaSuccess;
	FFT_Real *res = nullptr;
	cudaStatus = cudaMalloc((void **)&res, size);
	if (cudaStatus == cudaSuccess && res) {
		*ptr = res;
		return 1;
	}
	else {
		*ptr = nullptr;
		return 0;
	}
}

int allocateFFTRealExt(FFT_Real** ptr, int cols, int rows) {
	if (cols < 0) {
		*ptr = nullptr;
		return 0;
	}

	if (rows < 0) rows = 1;

	size_t size = static_cast<size_t>(cols)*static_cast<size_t>(rows)*sizeof(FFT_Real);

	cudaError_t cudaStatus = cudaSuccess;
	FFT_Real *res = nullptr;
	cudaStatus = cudaMalloc((void **)&res, size);
	if (cudaStatus == cudaSuccess && res) {
		*ptr = res;
		return 1;
	}
	else {
		*ptr = nullptr;
		return 0;
	}
}

int freeCudaMem(void *ptr) {
	cudaError_t cudaStatus = cudaSuccess;
	cudaStatus = cudaFree(ptr);
	if (cudaStatus == cudaSuccess) {
		return 1;
	}
	else {
		return 0;
	}
}

int cudaMemFromHost(void *dstDev, void *srcHost, size_t byteSize) {
	if (byteSize < 1) {
		return 0;
	}

	cudaError_t cudaStatus = cudaSuccess;
	cudaStatus = cudaMemcpy(dstDev, srcHost, byteSize, cudaMemcpyHostToDevice);
	if (cudaStatus == cudaSuccess) {
		return 1;
	}
	else {
		return 0;
	}

}

int cudaMemToHost(void *dstHost, void *srcDev, size_t byteSize) {
	if (byteSize < 1) {
		return 0;
	}

	cudaError_t cudaStatus = cudaSuccess;
	cudaStatus = cudaMemcpy(dstHost, srcDev, byteSize, cudaMemcpyDeviceToHost);
	if (cudaStatus == cudaSuccess) {
		return 1;
	}
	else {
		return 0;
	}

}

int cudaMemDevToDev(void *dstDev, void *srcDev, size_t byteSize) {
	if (byteSize < 1) {
		return 0;
	}

	cudaError_t cudaStatus = cudaSuccess;
	cudaStatus = cudaMemcpy(dstDev, srcDev, byteSize, cudaMemcpyDeviceToDevice);
	if (cudaStatus == cudaSuccess) {
		return 1;
	}
	else {
		return 0;
	}
}

int createFFTPlan1d_R2C(FFTPlan_Handle *plan, int cols, int rows) {
	if (cols < 0) {
		*plan = 0;
		return 0;
	}

	if (rows < 0) rows = 1;

	cufftResult cudaStatus = CUFFT_SUCCESS;
	cudaStatus = cufftPlan1d(plan, cols, CUFFT_R2C, rows);

	if (cudaStatus == CUFFT_SUCCESS) {
		return 1;
	}
	else {
		return 0;
	}
}

int destroyFFTPlan(FFTPlan_Handle plan) {
	cufftResult cudaStatus = CUFFT_SUCCESS;
	cudaStatus = cufftDestroy(plan);

	if (cudaStatus == CUFFT_SUCCESS) {
		return 1;
	}
	else {
		return 0;
	}
}

int execR2CfftPlan(FFTPlan_Handle plan, FFT_Real *idata, FFT_Complex *odata) {
	cufftResult cudaStatus = CUFFT_SUCCESS;
	cudaStatus = cufftExecR2C(plan, idata, (cufftComplex*)odata);
	if (cudaStatus == CUFFT_SUCCESS) {
		return 1;
	}
	else {
		return 0;
	}
}