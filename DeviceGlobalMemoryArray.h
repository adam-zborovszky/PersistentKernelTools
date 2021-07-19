#pragma once

#include "cuda_runtime.h"

/*

global memory array class
-------------------------
for device ONLY use!
- lifetime can span multiple kernel lauches
- allocates on device side ONLY
- avoid device side concurent access of the same elements!

*/

template<class item_t, class index_t>
class DeviceGlobalMemoryArray
{
private:
	item_t* _pointer;
	index_t _size;
public:
	DeviceGlobalMemoryArray::DeviceGlobalMemoryArray(index_t size);

	DeviceGlobalMemoryArray::~DeviceGlobalMemoryArray();

	item_t* Pointer();

	index_t Capacity();
};


