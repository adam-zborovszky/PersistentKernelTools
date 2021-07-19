#include "PinnedMemoryArray.h"

#include "cuda_runtime.h"


template<class item_t, class index_t>
PinnedMemoryArray<class item_t, class index_t>::PinnedMemoryArray(index_t size, bool isWriteCombined)
{
	_size = size;
	int flags = isWriteCombined ? cudaHostAllocWriteCombined : cudaHostAllocDefault;

	if (cudaHostAlloc(&_hostPointer, _size * sizeof(item_t), flags) != cudaSuccess)
	{
		_hostPointer = NULL;
		_size = 0;
	}
	if (cudaMalloc(&_devicePointer, _size * sizeof(item_t)) != cudaSuccess)
	{
		cudaFreeHost(_hostPointer);
		_devicePointer = NULL;
		_size = 0;
	}
}


template<class item_t, class index_t>
PinnedMemoryArray<class item_t, class index_t>::~PinnedMemoryArray()
{
	cudaFree(_devicePointer);
	cudaFreeHost(_hostPointer);
}


template<class item_t, class index_t>
item_t* PinnedMemoryArray<class item_t, class index_t>::HostPointer()
{
	return _hostPointer;
}


template<class item_t, class index_t>
item_t* PinnedMemoryArray<class item_t, class index_t>::DevicePointer()
{
	return _devicePointer;
}


template<class item_t, class index_t>
cudaError PinnedMemoryArray<class item_t, class index_t>::copyToHostAsync(cudaStream_t stream, cudaEvent_t event)
{
	cudaMemcpyAsync(_hostPointer, _devicePointer, _size * sizeof(item_t), cudaMemcpyDeviceToHost, stream);
	cudaEventRecord(event, stream);
	return error;
}


template<class item_t, class index_t>
cudaError PinnedMemoryArray<class item_t, class index_t>::copyToDeviceAsync(cudaStream_t stream, cudaEvent_t event)
{
	error = cudaMemcpyAsync(_devicePointer, _hostPointer, _size * sizeof(item_t), cudaMemcpyHostToDevice, stream);
	cudaEventRecord(event, stream);
	return error;
}