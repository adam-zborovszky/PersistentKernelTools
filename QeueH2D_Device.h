#pragma once

#include "cuda_runtime.h"
#include "QeueItemStatus.h"
#include "QeueHeader.h"
#include "PageAbleMemoryArray.h"
#include "MappedMemoryArray.h"
#include "PinnedMemoryArray.h"


/*

Device side instance of QeueH2D_Host
---------------------------------------------------

*/


template<class item_t>
class QeueH2D_Device
{
private:
	int _size; // maximum number of items in the qeue
	int _popFromHere; // next item index to push item into
	QeueItemStatus* _statuses; // array of item statuses
	item_t** _items; // array of item pointers
public:
	void Construct(int size, QeueItemStatus* statuses); // construct using the mapping pointer of statuses
	void Destruct();
	__device__ int GetItemForRead();
	__device__ bool SetItemToFree(int index);
	__device__ item_t* GetItemPointer(int itemIndex);
};