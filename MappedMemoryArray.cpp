#include "MappedMemoryArray.h"

#include "cuda_runtime.h"


template<class item_t, class index_t>
MappedMemoryArray<class item_t, class index_t>::MappedMemoryArray(index_t size)
{
	if (cudaHostAlloc(&_hostPointer, size * sizeof(item_t), cudaHostAllocMapped) != cudaSuccess)
	{
		_hostPointer = NULL;
		return;
	}

	if (cudaHostGetDevicePointer(&_devicePointer, _hostPointer, 0) != cudaSuccess)
	{
		cudaFreeHost(_hostPointer);
		_devicePointer = NULL;
		return;
	}
}


template<class item_t, class index_t>
MappedMemoryArray<class item_t, class index_t>::~MappedMemoryArray()
{
	cudaFreeHost(_hostPointer);
}


template<class item_t, class index_t>
item_t* MappedMemoryArray<class item_t, class index_t>::HostPointer()
{
	return _hostPointer;
}


template<class item_t, class index_t>
item_t* MappedMemoryArray<class item_t, class index_t>::DevicePointer()
{
	return _devicePointer;
}

