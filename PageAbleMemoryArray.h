#pragma once

#include "cuda_runtime.h"

/*

page-able memory array class
---------------------------------------
- slow because of double copy, but useful where data must be copied only ones during software run, because does not consume pinned memory area.

- host allocates in the host pinned memory area, device allocates in device global mem area
- transfer is initiated explicitly
- during copy all host data is transferred from host to device or from device to host
- the transfer happens in two copy steps
	- for the H2D copy the a pinned area is allocated by the system and page-able data are first transferred to there, then copied to the device
	- in case of D2H transfer the data is also going through a pinned memory area
	- the pinned area is allocated temporary for the copy process only
- transfer can happen during previous kernel calculations, so the transfer can be overlapped
- transfer latency: memory transfer time: 8.0 ms + kernel execution time: 0.5 ms

*/

template<class item_t, class index_t>
class PageAbleMemoryArray
{
private:
	item_t* _hostPointer;
	item_t* _devicePointer;
	index_t _size;
public:
	PageAbleMemoryArray(index_t size);

	~PageAbleMemoryArray();

	item_t* HostPointer();

	item_t* DevicePointer();

	cudaError copyToHost();

	cudaError copyToDevice();
};