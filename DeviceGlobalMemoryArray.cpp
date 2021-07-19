#include "DeviceGlobalMemoryArray.h"


template<class item_t, class index_t>
DeviceGlobalMemoryArray<class item_t, class index_t>::DeviceGlobalMemoryArray(index_t size)
{
	_size = size;
	if (cudaMalloc(&_pointer, _size * sizeof(item_t)) != cudaSuccess)
	{
		_pointer = NULL;
		_size = 0;
	}
		
}


template<class item_t, class index_t>
DeviceGlobalMemoryArray<class item_t, class index_t>::~DeviceGlobalMemoryArray()
{
	cudaFree(_pointer);
}


template<class item_t, class index_t>
item_t* DeviceGlobalMemoryArray<class item_t, class index_t>::Pointer()
{
	return _pointer;
}


template<class item_t, class index_t>
index_t DeviceGlobalMemoryArray<class item_t, class index_t>::Capacity()
{
	return _size;
}
