#include "QeueD2H.h"

#include "cuda_runtime.h"
#include "MappedMemoryArray.h"
#include "QeueItemStatus.h"


template<class item_t, class index_t>
QeueD2H_Host<class item_t, class index_t>::QeueD2H_Host(int capacity, index_t arraySize)
{
	_size = capacity;
	_arraySize = arraySize;
	_pushToHere = 0;
	_items = new MappedMemoryArray<class item_t, class index_t>[_size];
	for (int i; i < _size; i++)
		_items[i] = new MappedMemoryArray<class item_t, class index_t>(_arraySize)
		_statuses = new MappedMemoryArray(_size);
	for (int i = 0; i < _size; i++)
		_statuses.HostPointer[i] = QeueItemStatus::Free;
}


template<class item_t, class index_t>
QeueD2H_Host<class item_t, class index_t>::~QeueD2H_Host()
{
	for (int i; i < _size; i++)
		delete _items[i]; 
	delete _items;
	delete _statuses;
}


template<class item_t, class index_t>
int QeueD2H_Host<class item_t, class index_t>::GetItemForRead()
{
	index_t itemIndex = _popFromHere;
	itemStatusPointer = _statuses.HostPointer + _popFromHere;
	if (*itemStatusPointer == QeueItemStatus::Ready)
	{
		*itemStatusPointer = QeueItemStatus::ReadLocked;
		if (++_popFromHere == _size) _popFromHere = 0;
		return itemIndex;
	}
	else
		return -1; // empty
}


template<class item_t, class index_t>
bool QeueD2H_Host<class item_t, class index_t>::SetItemToFree(int itemIndex)
{
	itemStatusPointer = _statuses.HostPointer + itemIndex;
	if (*itemStatusPointer == QeueItemStatus::ReadLocked)
	{
		*itemStatusPointer = QeueItemStatus::Free;
		return true;
	}
	else
		return false; // the status was NOT ReadLocked
}


template<class item_t, class index_t>
item_t* QeueD2H_Host<class item_t, class index_t>::GetItemPointer(int itemIndex)
{
	return _items[itemIndex];
}


template<class item_t, class index_t>
__device__ QeueD2H_Device<class item_t, class index_t>::QeueD2H_Device(int capacity, index_t arraySize, item_t** arrays, QeueItemStatus* statuses)
{
	_size = capacity;
	_pushToHere = 0;
	_arraySize = arraySize;
	_items = arrays;
	_statuses = statuses;
}


template<class item_t, class index_t>
__device__ QeueD2H_Device<class item_t, class index_t>::~QeueD2H_Device()
{
}


template<class item_t, class index_t>
__device__ int QeueD2H_Device<class item_t, class index_t>::GetItemForWrite()
{
	QeueItemStatus readStatus;
	do
	{
		int itemIndex = _pushToHere;
		readStatus = atomicCAS(_statuses + itemIndex, QeueItemStatus::Free, QeueItemStatus::WriteLocked);
		if (readStatus == QeueItemStatus::Free) // push registering was succesful
		{
			atomicInc(&_pushToHere, _size); // increment (does not matter if threads increment in swapped order as actual value is used for access)
			return itemIndex;
		}
	} while (readStatus == QeueItemStatus::WriteLocked || pushToHereActual < _pushToHere) // maybe another thread successfully popped this item
		return -1; // the qeue is empty
}


template<class item_t, class index_t>
__device__ bool QeueD2H_Device<class item_t, class index_t>::SetItemToReady(int itemIndex)
{
	if (atomicCAS(_statuses + itemIndex, QeueItemStatus::WriteLocked, QeueItemStatus::Ready) == QeueItemStatus::WriteLocked)
		return true;
	else
		return false; // the status was NOT WriteLocked
}


template<class item_t, class index_t>
__device__ item_t* QeueD2H_Device<class item_t, class index_t>::GetItemPointer(int itemIndex)
{
	return _items[itemIndex];
}





