// general
#include <stdio.h>
#include <conio.h>
#include <assert.h> // error checking and error outputs
#include <iostream>
#include <math.h>
#include <stdio.h>
// #include <string>

#include "QeueH2D.h"
#include "QeueD2H.h"


// cuda
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

// opencv
#include "opencv2/opencv.hpp"



// GPU constant parameters
#define L1CacheSize 128 // bytes: global memory cache size - align global read/write to this granurality
#define L2CacheSize 32 // bytes: global memory cache size - align global read/write to this granurality
#define MappedMemoryGranurality 32 // officially not published, most probably 32 bytes per read / write, not caches





int main()
{
	int deviceNum, deviceId;
	cudaGetDeviceCount(&deviceNum);
	printf("Available devices = %d pc(s)\n\n", deviceNum);

	cudaDeviceProp prop;
	for (deviceId = 0; deviceId < deviceNum; deviceId++)
	{
		cudaGetDeviceProperties(&prop, deviceId);

		// print selected device capabilities
		printf("\n");
		printf("Device properties:\n\n");
		printf("    Device id                                   : %d\n", deviceId);
		printf("    Device name                                 : %s\n", prop.name);
		printf("    Compute capability                          : %d.%d\n", prop.major, prop.minor);
		printf("\n");
		printf("    Device parameters\n");
		printf("        number of SMs                           : %d\n", prop.multiProcessorCount);
		printf("        maximum threads per SM                  : %d\n", prop.maxThreadsPerMultiProcessor);
		printf("        maximum threads per block               : %d\n", prop.maxThreadsPerBlock);
		printf("        Total global mem                        : %llu MB\n", prop.totalGlobalMem / 1024 / 1024);
		printf("        available registers per SM              : %lu kB\n", prop.regsPerMultiprocessor / 1024);
		printf("        available shared mem per SM             : %llu kB\n", prop.sharedMemPerBlock / 1024);
		printf("        L2 cache size per SM                    : %d kB\n", prop.l2CacheSize / 1024);
	}

	// select device
	deviceId = 0;
	cudaSetDevice(deviceId);
	// reset device;
	cudaDeviceReset();

	printf("Selected device parameters:\n");
	cudaGetDeviceProperties(&prop, deviceId);

	// check hardware requirements

	printf("\n");
	printf("Device check:\n");
	// compiler settings: sm_61 compute_61
	assert(prop.major >= 6);
	assert(prop.minor >= 1);
	printf("    OK.\n");

	// GPU & kernel configuration

	// on architectures CC < 2.0 , where L1 cache is separated from shared mem : Larger shared memory and smaller L1 cache is preferred as shared mem stores activation values 
	// cudaDeviceSetCacheConfig(cudaFuncCachePreferShared); 

	// cudaDeviceMapHost = allow use of ZeroCopyMemory 
	cudaSetDeviceFlags(cudaDeviceMapHost);

	// calculate number of blocks and threads
	int blocksPerSM = prop.maxThreadsPerMultiProcessor / prop.maxThreadsPerBlock;
	dim3 blocks(prop.multiProcessorCount * blocksPerSM, 1, 1);
	dim3 threads(prop.warpSize, prop.maxThreadsPerBlock / prop.warpSize, 1); // thread array grupped by warps

	printf("\n");
	printf("Optimal persistent kernel configuration\n");
	printf("    SM data:\n");
	printf("        number of SMs                           : %d\n", prop.multiProcessorCount);
	printf("        blocks per SM                           : %d\n", blocksPerSM);
	printf("        (with this setting all blocks are active and shared mem is fragmented the least.\n");
	printf("        then total number of blocks             : %d\n", blocks.x);
	printf("\n");
	printf("    Block data:\n");
	printf("        warps per block                         : %d\n", threads.y);
	printf("        threads per block                       : %d\n", threads.x * threads.y);
	printf("        maximum shared mem per block            : %llu kB\n", prop.sharedMemPerBlock / blocksPerSM / 1024);
	printf("\n");
	printf("    Warp data:\n");
	printf("        threads per warp                        : %d\n", threads.x);
	printf("    Thread data:\n");
	printf("        average registers per thread            : %d\n", prop.regsPerBlock / blocksPerSM / threads.x / threads.y);
	printf("        average shared mem per thread           : %llu B\n", prop.sharedMemPerBlock / blocksPerSM / threads.x / threads.y);
	printf("\n");
	printf("    Others:\n");

	printf("        L1 cache transaction size               : %d B\n", L1CacheSize);
	printf("        L2 cache transaction size               : %d B (only L2 = uncached loads using the generic data path)\n", L2CacheSize);
	printf("        mapped memory mapping granurality       : %d B\n", MappedMemoryGranurality);
	printf("        texture cache transaction size          : %d B\n", 32);
	if (prop.major <= 2)
		printf("        L1 cache is separated from shared mem! Use cudaDeviceSetCacheConfig() !\n");
	if (prop.major >= 3)
		printf("        Shuffle available!\n");
	printf("\n");

	// create streams
	cudaStream_t computeStream;
	cudaStreamCreate(&computeStream);

	// input initialization

	// open 
	cv::VideoCapture camera(0);
	assert(camera.isOpened());

	// set videostream parameters
	camera.set(cv::CAP_PROP_FRAME_WIDTH, 10000); // max
	camera.set(cv::CAP_PROP_FRAME_HEIGHT, 10000); // max
	camera.set(cv::CAP_PROP_FPS, 10000); // max
	int inputFormat = CV_8UC3; // opencv format, unsigned 8 bit channels, total 3 channels: RGB 
	int inputChannels = 3; // CV_8UC3 = 3 x byte / pixel ,   B G R B G R B G R ...

	// read picture size
	int frameWidth = camera.get(cv::CAP_PROP_FRAME_WIDTH);
	int frameHeight = camera.get(cv::CAP_PROP_FRAME_HEIGHT);

	// display settings
	printf("\n");
	printf("Camera settings\n");
	printf("    frame size                           : %d x %d pixel\n", frameWidth, frameHeight);
	printf("    channels                             : %d \n", inputChannels);
	printf("    frame size in memory                 : %d kb\n", frameWidth * frameHeight * inputChannels / 1024);
	printf("    fps                                  : %d\n", int(camera.get(cv::CAP_PROP_FPS)));
	printf("\n");
	// input
	// 640 x 480 x RGB = 921600 byte = 0,88 MB
	// 1280 x 720 x RGB = 2764800 = 2,64 MB*
	// 4K x RGB = 26542080 byte = 25.3 MB


	QeueH2D_Host<unsigned char[36000], int> InputQeue(3);







	#define inputBufferSize 3
	inputDispatcherHost_t<unsigned char, int> inputDispatcher(inputBufferSize, frameWidth * frameHeight * inputChannels); // one frame
	cv::Mat* inputMat[inputBufferSize]; // buffer arrays will be wrapped into matrices one by one
	for (int i = 0; i < inputBufferSize; i++)
		inputMat[i] = new cv::Mat(cv::Size(frameWidth, frameHeight), inputFormat, inputDispatcher[i]);

	// output
	#define outputBufferSize 3
	outputDispatcherHost_t<unsigned char, int> outputDispatcher(outputBufferSize, frameWidth * frameHeight * inputChannels);
	cv::Mat* outputMat[outputBufferSize]; // buffer arrays will be wrapped into matrices one by one
	for (int i = 0; i < outputBufferSize; i++)
		outputMat[i] = new cv::Mat(cv::Size(frameWidth, frameHeight), inputFormat, outputDispatcher[i]);

	cv::namedWindow("output");


	// mem tarnsfer + kernel loop metrics timers
	clock_t startTimer, stopTimer;

	// collect kernel arguments
	void* arguments[] =
	{
		(size_t*)&inputDispatcher.arraySize,
		(void**)&inputDispatcher.deviceArray_d,
		&inputDispatcher.bufferSize,
		&inputDispatcher.hostStatus_d,
		&inputDispatcher.deviceStatus_d,
		(size_t*)&outputDispatcher.arraySize,
		(void**)&outputDispatcher.deviceArray_d,
		&outputDispatcher.bufferSize,
		&outputDispatcher.hostStatus_d,
		&outputDispatcher.deviceStatus_d
	};

	cudaError error;
	printf("host  : kernel launch...\n");
	error = cudaLaunchKernel(kernel, blocks, threads, arguments, 0, computeStream);
	printf("code  : %d, reason: %s\n", error, cudaGetErrorString(error));
	error = cudaStreamQuery(computeStream); // forces kernel launch, otherwise starts only after sync call (no problem if device is not ready)

	int inputId;
	int outputId;
	while (1)
	{
		if (cv::waitKey(1) == 27) break;
		while (inputDispatcher.get(&inputId)); // until get one input to fill
		stopTimer = clock();
		// printf("%3.0f msec \n", (stopTimer - startTimer) * 1000.0f / CLOCKS_PER_SEC);
		camera >> *inputMat[inputId];
		startTimer = clock();
		while (inputDispatcher.send());

		while (outputDispatcher.get(&outputId)); // until get one output to read
		cv::imshow("output", *outputMat[outputId]);
		while (outputDispatcher.send());

	}

	// emitting stop signals
	inputDispatcher.sendStop(totalBlocks);

	cv::destroyAllWindows();
	camera.release();
	//cudaEventDestroy(startEvent);
	//cudaEventDestroy(stopEvent);

	// free
	cudaStreamSynchronize(computeStream);

	return 0;
}