#pragma once

#include "cuda_runtime.h"

/*

mapped (zero copy) memory array class
---------------------------------------
- useful when kernel runs calculation intensive tasks, what hides the PCIe access latency, what is a magnitude bigger than global memory access latency

- host allocates in the host pinned memory area, what is mapped through the PCIe to the device side (not copied)
- device get data on the fly when kernel runs, latency can be hidden by calculations and global mem transactions only
- mapping happens in 32 byte chunks (non published parameter) - access must be optimized for this size (otherwise destroys performance)
- device memory is not used for this transaction type - can save device global memory
- mapped data arrives directly to the SMs, so no global memory acess is required
- obviously no GPU caching, but CPU caching happens above CUDA 9.x and if harware is capable
- transfer latency: memory transfer time: 0 ms if PCIe mapping is hidden + kernel execution time: 3.8 ms 
*/

template<class item_t, class index_t>
class MappedMemoryArray
{
private:
	item_t* _hostPointer;
	item_t* _devicePointer;
	index_t _size;
public:
	MappedMemoryArray(index_t _size);
	~MappedMemoryArray();
	item_t* HostPointer();
	item_t* DevicePointer();
};