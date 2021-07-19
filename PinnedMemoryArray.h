#pragma once

#include "cuda_runtime.h"

/*

pinned (page-locked) memory array class
---------------------------------------
- useful when data transfers can happen during parallel kernel calculation

- transfer is initiated explicitly
- host allocates in the host pinned memory area, device allocates in device global mem area
- during copy all host data is transferred to the device global memory
- transfers are async, and DMA can manage if available
- event is recorded when async copy has finished
- transfer can happen during previous kernel calculations, what can hide latency
- transfer latency: memory transfer time: 4.3 ms + kernel execution time: 0.5 ms

*/

template<class item_t, class index_t>
class PinnedMemoryArray
{
private:
	item_t* _hostPointer;
	item_t* _devicePointer;
	index_t _size;
public:
	// create a pinned (page-locked) allocation
	// if CPU writes only, then set isWriteCombined to true, but if reads (too), then set to false!
	PinnedMemoryArray(index_t size, bool isWriteCombined); 

	~PinnedMemoryArray();

	item_t* HostPointer();

	item_t* DevicePointer();

	cudaError copyToHostAsync(cudaStream_t _stream, cudaEvent_t _event);

	cudaError copyToDeviceAsync(cudaStream_t _stream, cudaEvent_t _event);
};