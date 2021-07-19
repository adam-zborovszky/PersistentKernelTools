


// cuda
#include "cuda_runtime.h"


// opencv
#include "opencv2/core.hpp"



// --------------------------------------------------------------------------------------------



#define gridLeader (!(threadIdx.x || threadIdx.y || blockIdx.x))
#define blockLeader (!(threadIdx.x || threadIdx.y ))

// grid level variables 
__device__ inputDispatcherDevice_t<unsigned char, int> inputDispatcher;
__device__ outputDispatcherDevice_t<unsigned char, int> outputDispatcher;
__device__ volatile unsigned int gridStatus = 0;



// device kernel
__global__ void kernel(
	size_t _inputArraySize,
	void** _inputArray,
	int _inputBufferSize,
	volatile int* _inputHostStatus,
	volatile int* _inputDeviceStatus,
	size_t _outputArraySize,
	void** _outputArray,
	int _outputBufferSize,
	volatile int* _outputHostStatus,
	volatile int* _outputDeviceStatus)
{
	// startup 
	if (gridLeader)
	{
		printf("device: kernel working.\n");
		inputDispatcher.setup(_inputArraySize, &_inputArray, _inputBufferSize, &_inputHostStatus, &_inputDeviceStatus);
		outputDispatcher.setup(_outputArraySize, &_outputArray, _outputBufferSize, &_outputHostStatus, &_outputDeviceStatus);
		gridStatus = gridDim.x; // number of blocks -> a signal for start & also required for exiting 
	}

	if (blockLeader)
	{
		while (gridStatus == 0); // wait for gridleader to startup
		printf("device: block[%d] running\n", blockIdx.x);
	}
	__syncthreads(); // threads must wait for block leaders

	while (1) // working loop
	{
		// block level variables, lifetime = working loop
		__shared__ volatile int inputId;
		__shared__ volatile int outputId;

		if (blockLeader) while (inputDispatcher.get((int*)&inputId)); // get new input
		__syncthreads(); // block level sync - to wait until the input is available for all the threads
		if (inputId == -1) break; // stop signal found
		if (blockLeader) while (outputDispatcher.get((int*)&outputId)); // get new output 


		// copy input data to shared or reg...
		int pixelP = 3 * (blockIdx.x * blockDim.x * blockDim.y + threadIdx.y * warpSize + threadIdx.x);
		unsigned char* in = inputDispatcher[inputId];
		unsigned char c0 = in[pixelP + 0];
		unsigned char c1 = in[pixelP + 1];
		unsigned char c2 = in[pixelP + 2];

		// copy shared data to output ...
		unsigned char* out = outputDispatcher[outputId];
		out[pixelP + 0] = c0;
		out[pixelP + 1] = c1;
		out[pixelP + 2] = c2;

		__syncthreads();

		if (blockLeader)
		{
			while (inputDispatcher.send((int*)&inputId)); // send back (free) input - no effect in the first round
			while (outputDispatcher.send((int*)&outputId)); // send output - no effect in the first round
		}
	}

	if (gridLeader) while (gridStatus > 1); // wait for the other blockleaders, until they exit
	if (blockLeader) printf("device: block[%d] status was = %d\n", blockIdx.x, atomicDec((unsigned int*)&gridStatus, 1));
	__syncthreads();
	if (blockLeader) printf("device: block[%d] exiting\n", blockIdx.x);

}



