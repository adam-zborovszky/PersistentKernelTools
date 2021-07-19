#include "PageAbleMemoryArray.h"

#include "cuda_runtime.h"


template<class item_t, class index_t>
PageAbleMemoryArray<class item_t, class index_t>::PageAbleMemoryArray(index_t size)
{
	_size = size;
	_hostPointer = malloc(_size * sizeof(item_t);
	if (cudaMalloc(&_devicePointer, _size * sizeof(item_t)) != cudaSuccess)
	{
		free(_hostPointer);
		_devicePointer = NULL;
		_size = 0;
	}
}


template<class item_t, class index_t>
PageAbleMemoryArray<class item_t, class index_t>::~PageAbleMemoryArray()
{
	cudaFree(_devicePointer);
	free(_hostPointer);
}


template<class item_t, class index_t>
item_t* PageAbleMemoryArray<class item_t, class index_t>::HostPointer()
{
	return _hostPointer;
}


template<class item_t, class index_t>
item_t* PageAbleMemoryArray<class item_t, class index_t>::DevicePointer()
{
	return _devicePointer;
}


template<class item_t, class index_t>
cudaError PageAbleMemoryArray<class item_t, class index_t>::copyToHost()
{
	return cudaMemcpy(_hostPointer, _devicePointer, _size * sizeof(item_t), cudaMemcpyDeviceToHost);
}


template<class item_t, class index_t>
cudaError PageAbleMemoryArray<class item_t, class index_t>::copyToDevice()
{
	return cudaMemcpy(_devicePointer, _hostPointer, _size * sizeof(item_t), cudaMemcpyHostToDevice);
}