#pragma once

#include "cuda_runtime.h"

/*

managed (unified) memory class
-------------------------------------
- pointer is accessible on host & device side transparently
- slower than other types, mainly if concurent host and device access happens
- transfer latency: memory transfer time: 2.0 ms + kernel execution time: 16.1 ms

*/

template<class item_t, class index_t>
class ManagedMemoryArray
{
private:
	item_t* _pointer;
	index_t _size;
public:
	ManagedMemoryArray(index_t size);

	~ManagedMemoryArray();

	item_t* Pointer();
};



