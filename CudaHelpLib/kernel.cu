
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cufft.h"

#include <stdio.h>

#include "CudaHelpLib.h"

static int isCudaOk = 0;
static _CUDA_DEV_INFO deviceinfo;

///dev constant define
__constant__  int devC_cols;
__constant__  int devC_rows;
__constant__  int devC_divc;
__constant__  int devC_x;
__constant__  int devC_y;
__constant__  float devC_f1;
__constant__  float devC_xe;
__constant__  float devC_ye;
__constant__  float devC_ze;
__constant__ unsigned int devC_Palette[512];
//float guassianTable[512];

////////////////////

static int *dev_temp_4M1 = 0;
static int *dev_temp_4M2 = 0;
static int *dev_temp_4M3 = 0;
static unsigned char *dev_background_4M = 0;
static unsigned char *dev_cuboid = 0;
////////////////////////////////



//>>>>>>>>>>>>>>>>share lib main func
#ifndef Q_OS_LINUX
#include "Windows.h"

BOOL APIENTRY DllMain(HMODULE hModule,
	DWORD  ul_reason_for_call,
	LPVOID lpReserved
	)
{
	switch (ul_reason_for_call)
	{
	case DLL_PROCESS_ATTACH:
		break;
	case DLL_THREAD_ATTACH:
		break;
	case DLL_THREAD_DETACH:
		break;
	case DLL_PROCESS_DETACH:
		CuH_FreeTempCudaMem();
		break;
	}
	return TRUE;
}

#endif
//<<<<<<<<<<<<<


//////////////////////////////////////////////////////////////////////////
// >>>>>>>>>>>>>>>>>>>>>>>>>>>>>cuda  kernel function //////////////////
//////////////////////////////////////////////////////////////////////

__global__ void magnitude32F_Kernel(FFT_Complex * datain, FFT_Real * dataout) {
	const unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
	const unsigned int j = blockIdx.y*blockDim.y + threadIdx.y;
	if (i<devC_cols && j<devC_rows) {
		const int index = j*devC_cols + i;
		float d = datain[index].re*datain[index].re + datain[index].im*datain[index].im;
		d = sqrtf(d);
		dataout[index] = d;
	}
}

__global__ void logAddBeta32F_Kernel(FFT_Real * datain, FFT_Real * dataout) {
	const unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
	const unsigned int j = blockIdx.y*blockDim.y + threadIdx.y;
	if (i<devC_cols && j<devC_rows) {
		const int index = j*devC_cols + i;
		float beta = *((float*)(&devC_y));
		float d = datain[index];
		d = d + beta;
		d = logf(d);
		dataout[index] = d;
	}
}

__global__ void cvtAndScale32Fto16U_Kernel(FFT_Real * datain, unsigned short * dataout) {
	const unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
	const unsigned int j = blockIdx.y*blockDim.y + threadIdx.y;
	if (i<devC_cols && j<devC_rows) {
		const int index = j*devC_cols + i;
		float d = *((float*)(&devC_x));
		float beta = *((float*)(&devC_y));
		d = d*datain[index] + beta;
		d = (d >= 0 && d <= 65535.0f)*d + (d > 65535.0f)*65535.0f;
		dataout[index] = d;
	}
}

__global__ void cpyRealToComplex_Kernel(FFT_Real * datain, FFT_Complex * dataout) {
	const unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
	const unsigned int j = blockIdx.y*blockDim.y + threadIdx.y;
	if (i<devC_cols && j<devC_rows) {
		const int index = j*devC_cols + i;
		dataout[index].re = datain[index];
		dataout[index].im = 0;
	}
}

__global__ void cpy16UC1ToComplex_Kernel(unsigned short * datain, FFT_Complex * dataout) {
	const unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
	const unsigned int j = blockIdx.y*blockDim.y + threadIdx.y;
	if (i<devC_cols && j<devC_rows) {
		const int index = j*devC_cols + i;
		dataout[index].re = datain[index];
		dataout[index].im = 0;
	}
}

__global__ void ROI_Complex_Kernel(FFT_Complex * datain, FFT_Complex *dataout) {
	unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
	unsigned int j = blockIdx.y*blockDim.y + threadIdx.y;
	if (i<devC_cols && j<devC_rows) {
		dataout[j*devC_cols + i].re = datain[(j + devC_y)*devC_divc + i + devC_x].re;
		dataout[j*devC_cols + i].im = datain[(j + devC_y)*devC_divc + i + devC_x].im;
	}
}

__global__ void transpose16UC1_Kernel(unsigned short * datain, unsigned short *dataout) {
	const unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
	const unsigned int j = blockIdx.y*blockDim.y + threadIdx.y;
	unsigned char data;
	if (i<devC_cols && j<devC_rows) {
		dataout[i*devC_rows + j] = datain[j*devC_cols + i];
	}
}

//////////////////////////////////////////////////////////////////////////
// <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<cuda  kernel function //////////////////
//////////////////////////////////////////////////////////////////////


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

void CUDACALLMODE cudaNop(void) {
	cudaError_t cudaStatus = cudaSuccess;

	if (dev_temp_4M1 == 0) {
		cudaStatus = cudaMalloc((void**)&dev_temp_4M1, 1024 * 1024 * 32);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMalloc failed!");

		}
	}

	if (dev_temp_4M2 == 0) {
		cudaStatus = cudaMalloc((void**)&dev_temp_4M2, 1024 * 1024 * 32);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMalloc failed!");

		}
	}
	if (dev_temp_4M3 == 0) {
		cudaStatus = cudaMalloc((void**)&dev_temp_4M3, 1024 * 1024 * 32);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMalloc failed!");

		}
	}
	if (dev_background_4M == 0) {
		cudaStatus = cudaMalloc((void**)&dev_background_4M, 1024 * 1024 * 32);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMalloc failed!");

		}
	}
}

void CuH_FreeTempCudaMem(void) {
	if (dev_temp_4M1 != 0) {
		cudaFree(dev_temp_4M1);
		dev_temp_4M1 = 0;
	}

	if (dev_temp_4M2 != 0) {
		cudaFree(dev_temp_4M2);
		dev_temp_4M2 = 0;
	}

	if (dev_temp_4M3 != 0) {
		cudaFree(dev_temp_4M2);
		dev_temp_4M3 = 0;
	}

	if (dev_background_4M != 0) {
		cudaFree(dev_background_4M);
		dev_background_4M = 0;
	}


	if (cudaDeviceSynchronize() != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error!\n");
	}


	if (cudaDeviceReset() != cudaSuccess) {
		fprintf(stderr, "cudaDeviceReset failed!\n");
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

int createFFTPlan1d_C2C(FFTPlan_Handle *plan, int cols, int rows) {
	if (cols < 0) {
		*plan = 0;
		return 0;
	}

	if (rows < 0) rows = 1;

	cufftResult cudaStatus = CUFFT_SUCCESS;
	cudaStatus = cufftPlan1d(plan, cols, CUFFT_C2C, rows);

	if (cudaStatus == CUFFT_SUCCESS) {
		return 1;
	}
	else {
		return 0;
	}
}

int createFFTPlan1d_C2R(FFTPlan_Handle *plan, int cols, int rows) {
	if (cols < 0) {
		*plan = 0;
		return 0;
	}

	if (rows < 0) rows = 1;

	cufftResult cudaStatus = CUFFT_SUCCESS;
	cudaStatus = cufftPlan1d(plan, cols, CUFFT_C2R, rows);

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

int execC2CfftPlan(FFTPlan_Handle plan, FFT_Complex *idata, FFT_Complex *odata ,int dir) {
	cufftResult cudaStatus = CUFFT_SUCCESS;
	int d = CUFFT_FORWARD;
	if (d) {
		d = CUFFT_INVERSE;
	}
	cudaStatus = cufftExecC2C(plan, (cufftComplex*)idata, (cufftComplex*)odata, d);
	if (cudaStatus == CUFFT_SUCCESS) {
		return 1;
	}
	else {
		return 0;
	}
}

int execC2RfftPlan(FFTPlan_Handle plan, FFT_Complex *idata, FFT_Real *odata) {
	cufftResult cudaStatus = CUFFT_SUCCESS;
	cudaStatus = cufftExecC2R(plan,(cufftComplex *) idata, (cufftReal*)odata);
	if (cudaStatus == CUFFT_SUCCESS) {
		return 1;
	}
	else {
		return 0;
	}
}

int CuH_magnitudeDevC2R(FFT_Complex *devSrc, int cols, int rows, FFT_Real *hostDst) {
	cudaError_t cudaStatus = cudaSuccess;
	if (dev_temp_4M1 == 0 || dev_temp_4M2 == 0) {
		printf("cuda mem alloc faild.\n");
		return 1;
	}

	cudaStatus = cudaMemcpyToSymbol(devC_rows, &rows, sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "set const var failed!");
		return 1;
	}

	cudaStatus = cudaMemcpyToSymbol(devC_cols, &cols, sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "set const var failed!");
		return 1;
	}

	//calc block size
	dim3 blockS(16, 16);
	dim3 gridS(cols / 16, rows / 16);
	if (cols % 16) {
		gridS.x += 1;
	}
	if (rows % 16) {
		gridS.y += 1;
	}

	FFT_Complex *srcPtr = 0;
	if (devSrc) {
		srcPtr = devSrc;
	}
	else {
		srcPtr = (FFT_Complex*)dev_temp_4M1;
		cudaStatus = cudaMemcpy(dev_temp_4M1, dev_temp_4M2, rows*cols*sizeof(FFT_Complex), cudaMemcpyDeviceToDevice);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMemcpy failed!");
			return 1;
		}
	}

	magnitude32F_Kernel <<<gridS, blockS >>>(srcPtr, (FFT_Real*)dev_temp_4M2);
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "magnitude32F_Kernel Kernel failed: %s\n", cudaGetErrorString(cudaStatus));
		return 1;
	}

	//wait kernel finish
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "CuH_magnitudeDevC2R returned error code %d after launching magnitude32F_Kernel!\n", cudaStatus);
		return 1;
	}

	if (hostDst) {
		cudaStatus = cudaMemcpy(hostDst, dev_temp_4M2, rows*cols*sizeof(FFT_Real), cudaMemcpyDeviceToHost);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMemcpy failed!");
			return 1;
		}
	}

	return 0;
}


int CuH_logDevR2R(FFT_Real *devSrc, int cols, int rows, float beta, FFT_Real *hostDst) {
	cudaError_t cudaStatus = cudaSuccess;
	if (dev_temp_4M1 == 0 || dev_temp_4M2 == 0) {
		printf("cuda mem alloc faild.\n");
		return 1;
	}

	cudaStatus = cudaMemcpyToSymbol(devC_rows, &rows, sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "set const var failed!");
		return 1;
	}

	cudaStatus = cudaMemcpyToSymbol(devC_cols, &cols, sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "set const var failed!");
		return 1;
	}

	cudaStatus = cudaMemcpyToSymbol(devC_y, &beta, sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "set const var failed!");
		return 1;
	}

	//calc block size
	dim3 blockS(16, 16);
	dim3 gridS(cols / 16, rows / 16);
	if (cols % 16) {
		gridS.x += 1;
	}
	if (rows % 16) {
		gridS.y += 1;
	}

	FFT_Real *srcPtr = 0;
	if (devSrc) {
		srcPtr = devSrc;
	}
	else {
		srcPtr = (FFT_Real*)dev_temp_4M1;
		cudaStatus = cudaMemcpy(dev_temp_4M1, dev_temp_4M2, rows*cols*sizeof(FFT_Real), cudaMemcpyDeviceToDevice);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMemcpy failed!");
			return 1;
		}
	}

	logAddBeta32F_Kernel <<<gridS, blockS >>>(srcPtr, (FFT_Real*)dev_temp_4M2);
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "logAddBeta32F_Kernel Kernel failed: %s\n", cudaGetErrorString(cudaStatus));
		return 1;
	}

	//wait kernel finish
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "CuH_logDevR2R returned error code %d after launching logAddBeta32F_Kernel!\n", cudaStatus);
		return 1;
	}

	if (hostDst) {
		cudaStatus = cudaMemcpy(hostDst, dev_temp_4M2, rows*cols*sizeof(FFT_Real), cudaMemcpyDeviceToHost);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMemcpy failed!");
			return 1;
		}
	}

	return 0;
}

int CuH_cvtDevRealTo16UC1(FFT_Real *devSrc, int cols, int rows, float alpha, float beta, unsigned short *hostDst) {
	cudaError_t cudaStatus = cudaSuccess;
	if (dev_temp_4M1 == 0 || dev_temp_4M2 == 0) {
		printf("cuda mem alloc faild.\n");
		return 1;
	}

	cudaStatus = cudaMemcpyToSymbol(devC_rows, &rows, sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "set const var failed!");
		return 1;
	}

	cudaStatus = cudaMemcpyToSymbol(devC_cols, &cols, sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "set const var failed!");
		return 1;
	}

	cudaStatus = cudaMemcpyToSymbol(devC_x, &alpha, sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "set const var failed!");
		return 1;
	}

	cudaStatus = cudaMemcpyToSymbol(devC_y, &beta, sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "set const var failed!");
		return 1;
	}

	//calc block size
	dim3 blockS(16, 16);
	dim3 gridS(cols / 16, rows / 16);
	if (cols % 16) {
		gridS.x += 1;
	}
	if (rows % 16) {
		gridS.y += 1;
	}

	FFT_Real *srcPtr = 0;
	if (devSrc) {
		srcPtr = devSrc;
	}
	else {
		srcPtr = (FFT_Real*)dev_temp_4M1;
		cudaStatus = cudaMemcpy(dev_temp_4M1, dev_temp_4M2, rows*cols*sizeof(FFT_Real), cudaMemcpyDeviceToDevice);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMemcpy failed!");
			return 1;
		}
	}

	cvtAndScale32Fto16U_Kernel <<<gridS, blockS >>>(srcPtr, (unsigned short*)dev_temp_4M2);
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cvtAndScale32Fto16U_Kernel Kernel failed: %s\n", cudaGetErrorString(cudaStatus));
		return 1;
	}

	//wait kernel finish
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "CuH_cvtDevRealTo16UC1 returned error code %d after launching cvtAndScale32Fto16U_Kernel!\n", cudaStatus);
		return 1;
	}

	if (hostDst) {
		cudaStatus = cudaMemcpy(hostDst, dev_temp_4M2, rows*cols*sizeof(unsigned short), cudaMemcpyDeviceToHost);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMemcpy failed!");
			return 1;
		}
	}

	return 0;
}


int CuH_cpyHostRealToDevComplex(FFT_Real *srcHost, FFT_Complex *dstDev, int cols, int rows) {
	if (!dstDev) return 1;
	
	cudaError_t cudaStatus = cudaSuccess;
	if (dev_temp_4M1 == 0 || dev_temp_4M2 == 0) {
		printf("cuda mem alloc faild.\n");
		return 1;
	}

	cudaStatus = cudaMemcpyToSymbol(devC_rows, &rows, sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "set const var failed!");
		return 1;
	}

	cudaStatus = cudaMemcpyToSymbol(devC_cols, &cols, sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "set const var failed!");
		return 1;
	}

	
	//calc block size
	dim3 blockS(16, 16);
	dim3 gridS(cols / 16, rows / 16);
	if (cols % 16) {
		gridS.x += 1;
	}
	if (rows % 16) {
		gridS.y += 1;
	}

	if (srcHost) {
		cudaStatus = cudaMemcpy(dev_temp_4M1, srcHost, rows*cols*sizeof(FFT_Real), cudaMemcpyHostToDevice);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMemcpy failed!");
			return 1;
		}
	}
	else {
		cudaStatus = cudaMemcpy(dev_temp_4M1, dev_temp_4M2, rows*cols*sizeof(FFT_Real), cudaMemcpyDeviceToDevice);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMemcpy failed!");
			return 1;
		}
	}
	
	cpyRealToComplex_Kernel <<<gridS, blockS >>>((FFT_Real*)dev_temp_4M1, dstDev);
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cpyRealToComplex_Kernel Kernel failed: %s\n", cudaGetErrorString(cudaStatus));
		return 1;
	}
	
	//wait kernel finish
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "CuH_cpyHostRealToDevComplex returned error code %d after launching cpyRealToComplex_Kernel!\n", cudaStatus);
		return 1;
	}

	
		//cudaStatus = cudaMemcpy(dev_temp_4M2, dstDev, rows*cols*sizeof(FFT_Complex), cudaMemcpyDeviceToDevice);
		//if (cudaStatus != cudaSuccess) {
		//	fprintf(stderr, "cudaMemcpy failed!");
		//	return 1;
		//}


	return 0;

}

int CuH_cpy16UC1ToDevComplex(unsigned short *srcHost, FFT_Complex *dstDev, int cols, int rows) {
	if (!dstDev) return 1;

	cudaError_t cudaStatus = cudaSuccess;
	if (dev_temp_4M1 == 0 || dev_temp_4M2 == 0) {
		printf("cuda mem alloc faild.\n");
		return 1;
	}

	cudaStatus = cudaMemcpyToSymbol(devC_rows, &rows, sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "set const var failed!");
		return 1;
	}

	cudaStatus = cudaMemcpyToSymbol(devC_cols, &cols, sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "set const var failed!");
		return 1;
	}


	//calc block size
	dim3 blockS(16, 16);
	dim3 gridS(cols / 16, rows / 16);
	if (cols % 16) {
		gridS.x += 1;
	}
	if (rows % 16) {
		gridS.y += 1;
	}

	if (srcHost) {
		cudaStatus = cudaMemcpy(dev_temp_4M1, srcHost, rows*cols*sizeof(unsigned short), cudaMemcpyHostToDevice);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMemcpy failed!");
			return 1;
		}
	}
	else {
		cudaStatus = cudaMemcpy(dev_temp_4M1, dev_temp_4M2, rows*cols*sizeof(unsigned short), cudaMemcpyDeviceToDevice);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMemcpy failed!");
			return 1;
		}
	}

	cpy16UC1ToComplex_Kernel <<<gridS, blockS >>>((unsigned short*)dev_temp_4M1, dstDev);
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cpy16UC1ToComplex_Kernel Kernel failed: %s\n", cudaGetErrorString(cudaStatus));
		return 1;
	}

	//wait kernel finish
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "CuH_cpyHostRealToDevComplex returned error code %d after launching cpy16UC1ToComplex_Kernel!\n", cudaStatus);
		return 1;
	}


	//cudaStatus = cudaMemcpy(dev_temp_4M2, dstDev, rows*cols*sizeof(FFT_Complex), cudaMemcpyDeviceToDevice);
	//if (cudaStatus != cudaSuccess) {
	//	fprintf(stderr, "cudaMemcpy failed!");
	//	return 1;
	//}


	return 0;
}

int CuH_ROIdevComplex(FFT_Complex *dataDev, int cols, int rows, int x, int y, int width, int height) {

	cudaError_t cudaStatus = cudaSuccess;
	if (dev_temp_4M1 == 0 || dev_temp_4M2 == 0) {
		printf("cuda mem alloc faild.\n");
		return 1;
	}

	cudaStatus = cudaMemcpyToSymbol(devC_x, &x, sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "set const var failed!");
		return 1;
	}

	cudaStatus = cudaMemcpyToSymbol(devC_y, &y, sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "set const var failed!");
		return 1;
	}

	cudaStatus = cudaMemcpyToSymbol(devC_cols, &width, sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "set const var failed!");
		return 1;
	}

	cudaStatus = cudaMemcpyToSymbol(devC_rows, &height, sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "set const var failed!");
		return 1;
	}

	cudaStatus = cudaMemcpyToSymbol(devC_divc, &cols, sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "set const var failed!");
		return 1;
	}

	FFT_Complex *srcCptr = nullptr;
	if (dataDev) {
		srcCptr = dataDev;
	}
	else {
		srcCptr = (FFT_Complex *)dev_temp_4M2;
	}
	cudaStatus = cudaMemcpy((void*)dev_temp_4M1, (void*)srcCptr, rows*cols*sizeof(FFT_Complex), cudaMemcpyDeviceToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		return 1;
	}


	//calc block size
	dim3 blockS(16, 16);
	dim3 gridS(width / 16, height / 16);
	if (width % 16) {
		gridS.x += 1;
	}
	if (height % 16) {
		gridS.y += 1;
	}

	ROI_Complex_Kernel<<<gridS, blockS >>>((FFT_Complex*)dev_temp_4M1, srcCptr);
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "ROI_Complex_Kernel Kernel failed: %s\n", cudaGetErrorString(cudaStatus));
		return 1;
	}

	//wait kernel finish
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching ROI_Complex_Kernel!\n", cudaStatus);
		return 1;
	}

	return  0;

}

int  transpose16UC1(int rows, int cols, void* dev_src, void *dev_dst)
{
	int res = 0;
	cudaError_t cudaStatus = cudaSuccess;
	//copy constant

	cudaStatus = cudaMemcpyToSymbol(devC_cols, &cols, sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "set const var failed!");
		return 1;
	}

	cudaStatus = cudaMemcpyToSymbol(devC_rows, &rows, sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "set const var failed!");
		return 1;
	}
	//calc block size
	dim3 blockS(16, 16);
	dim3 gridS(cols / 16, rows / 16);
	if (cols % 16) {
		gridS.x += 1;
	}
	if (rows % 16) {
		gridS.y += 1;
	}


	//invoke kernel
	transpose16UC1_Kernel <<<gridS, blockS >>>((unsigned short *)dev_src, (unsigned short *)dev_dst);
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "transpose16UC1 Kernel failed: %s\n", cudaGetErrorString(cudaStatus));
		return 1;
	}

	//wait kernel finish
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching transpose16UC1_Kernel!\n", cudaStatus);
		return 1;
	}

	return res;
}

int CuH_transpose16UC1(int rows, int cols, void* host_src, void *host_dst) {
	cudaError_t cudaStatus = cudaSuccess;

	if (dev_temp_4M1 == 0 || dev_temp_4M2 == 0) {
		printf("cuda mem alloc faild.\n");
		return 1;
	}

	if (host_src) {
		cudaStatus = cudaMemcpy(dev_temp_4M1, host_src, rows*cols*sizeof(unsigned short), cudaMemcpyHostToDevice);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMemcpy failed!");
			return 1;
		}
	}
	else {
		cudaStatus = cudaMemcpy(dev_temp_4M1, dev_temp_4M2, rows*cols*sizeof(unsigned short), cudaMemcpyDeviceToDevice);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMemcpy failed!");
			return 1;
		}
	}

	if (transpose16UC1(rows, cols, (void*)dev_temp_4M1, (void*)dev_temp_4M2)) return 1;

	if (host_dst) {
		cudaStatus = cudaMemcpy(host_dst, dev_temp_4M2, rows*cols*sizeof(unsigned short), cudaMemcpyDeviceToHost);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMemcpy failed!");
			return 1;
		}
	}

	return  0;
}