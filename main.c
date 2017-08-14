
#include <stdio.h>
#include <stdlib.h>
#include <cufft.h>

#define DATA_LENGTH 256

#ifdef __CUDACC__
typedef float2 fComplex;
#else
typedef struct
{
  float x;
  float y;
} fComplex;
#endif

void loadInputData(float *array){
  int i;
  FILE * fp;
  fp = fopen("input.txt","r");
  for(i=0; i<DATA_LENGTH; i++){
    fscanf(fp,"%f\n",array+i);
  }
}

void storeOutputData(fComplex *array){
  int i;
  FILE * fp;
  fp = fopen("output.txt","w");
  for(i=0; i<DATA_LENGTH; i++){
    fprintf(fp,"%f\t%f\n",*(array+2*i),*(array+2*i+1));
  }
}

#define cudaErrorCheck(error) __cudaErrorCheck(error, __LINE__)

void __cudaErrorCheck(cudaError_t error, const int line){
  if(error != cudaSuccess){
    printf("CUDA error at line %i\n", line);
    exit(-1);
  }
}


int main(void){

  // Create a "plan" object.
  cufftHandle fft_plan;

  float *host_signal;
  fComplex *host_spectrum;
  float *device_signal;
  fComplex *device_spectrum;

  // Allocate memeory on the host for the signal and spectrum data.
  host_signal = (float *)malloc(DATA_LENGTH*sizeof(float));
  host_spectrum = (fComplex *)malloc((DATA_LENGTH+1)*sizeof(fComplex));

  // Allocate memory on the device for the signal and spectrum data.
  cudaErrorCheck(cudaMalloc((void **)&device_signal, DATA_LENGTH*sizeof(float)));
  cudaErrorCheck(cudaMalloc((void **)&device_spectrum, (DATA_LENGTH+1)*sizeof(fComplex)));

  // Load the signal data into the host memory.
  loadInputData(host_signal);

  // Copy the signal data from the host to the device.
  cudaErrorCheck(cudaMemcpy(device_signal, host_signal, DATA_LENGTH*sizeof(float), cudaMemcpyHostToDevice));

  // Configure the "plan" object.
  cudaErrorCheck(cufftPlan1d(&fft_plan, DATA_LENGTH, CUFFT_R2C, 1));

  // Run the fft.
  cudaErrorCheck(cufftExecR2C(fft_plan, (cufftReal *)device_signal, (cufftComplex *)device_spectrum));

  // Make sure the fft completes.
  cudaErrorCheck(cudaDeviceSynchronize());

  // Copy the frequency data back over to the host.
  cudaErrorCheck(cudaMemcpy(host_spectrum, device_spectrum, (DATA_LENGTH+1)*sizeof(fComplex), cudaMemcpyDeviceToHost));

  // Write the spectrum data to a file.
  storeOutputData(host_spectrum);

  // Free memory on both the device and the host.
  cudaErrorCheck(cudaFree(device_signal));
  cudaErrorCheck(cudaFree(device_spectrum));

  free(host_signal);
  free(host_spectrum);

  return 0;
}
