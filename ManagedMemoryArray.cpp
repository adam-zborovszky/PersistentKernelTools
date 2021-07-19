#include "ManagedMemoryArray.h"


template<class item_t, class index_t>
ManagedMemoryArray<class item_t, class index_t>::ManagedMemoryArray(index_t size)
{
	_size = size;
	if (cudaMallocManaged(&_pointer, _size * sizeof(item_t)) != cudaSuccess)
	{
		_pointer = NULL;
		_size = 0;
	}
}


template<class item_t, class index_t>
ManagedMemoryArray<class item_t, class index_t>::~ManagedMemoryArray()
{
	cudaFree((void*)_pointer);
}


template<class item_t, class index_t>
item_t* ManagedMemoryArray<class item_t, class index_t>::Pointer()
{
	return _pointer;
}