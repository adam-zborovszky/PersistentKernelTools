#include "QeueH2D_Device.h"

#include "cuda_runtime.h"
#include "CudaCheckErrors.h"
#include "QeueItemStatus.h"
#include "QeueHeader.h"
#include "MappedMemoryArray.h"


template<class item_t>
void QeueH2D_Device<class item_t>::Construct(int size, QeueItemStatus* statuses)
{
	_size = size;
	_popFromHere = 0;
	_statuses = statuses;
		
	CudaCheckErrors(cudaMalloc(&_items, _size * sizeof(item_t*))); // allocate array of item pointers
	for (int i = 0; i < _size; i++)
		CudaCheckErrors(cudaMalloc(&(_items[i]), sizeof(item_t))); // allocate items
}


template<class item_t>
void QeueH2D_Device<class item_t>::Destruct()
{
	for (int i = 0; i < _size; i++)
		cudaFree(_items[i]);
	cudaFree(_items);
	// this class contains only a mapping pointer to statuses, the allocation is handled in the device qeue class
}


template<class item_t>
__device__ int QeueH2D_Device<class item_t>::GetItemForRead()
{
	QeueItemStatus readStatus;
	do
	{
		int itemIndex = _popFromHere;
		readStatus = atomicCAS(_statuses + itemIndex, QeueItemStatus::Ready, QeueItemStatus::ReadLocked);
		if (readStatus == QeueItemStatus::Ready) // pop registering was succesful
		{
			atomicInc(&_popFromHere, _size); // increment (does not matter if threads increment in swapped order as actual value is used for access)
			return itemIndex;
		}
	} while (readStatus == QeueItemStatus::ReadLocked || popFromHereActual < _popFromHere) // maybe another thread successfully popped this item
		return -1; // the qeue is empty
}


template<class item_t>
__device__ bool QeueH2D_Device<class item_t>::SetItemToFree(int index)
{
	if (atomicCAS(_statuses + index, QeueItemStatus::ReadLocked, QeueItemStatus::Free) == QeueItemStatus::ReadLocked)
		return true;
	else
		return false; // the status was NOT ReadLocked
}


template<class item_t>
__device__ item_t* QeueH2D_Device<class item_t>::GetItemPointer(int itemIndex)
{
	return _items[itemIndex];
}






