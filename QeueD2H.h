#pragma once

#include "cuda_runtime.h"
#include "QeueItemStatus.h"
#include "QeueHeader.h"
#include "MappedMemoryArray.h"
#include "PinnedMemoryArray.h"


/*

One-directional, Device -> Host data transfer queue
---------------------------------------------------
Manages an array of items, what can be written on device side and read on host side
- contains a read/write locking mechanism
- creates a device class for device side functions
- racing conditions are handled on device side only, so for multiple CPU threads use separate qeues!

*/


template<class item_t, class index_t>
class QeueD2H_Host
{
private:
	int _size;
	int _popFromHere;
	index_t _arraySize;
	MappedMemoryArray<QeueItemStatus, int> _statuses;
	PinnedMemoryArray<item_t, index_t> _items[];

public:
	QeueD2H_Host(int capacity, index_t arraySize);
	~QeueD2H_Host();
	int GetItemForRead();
	bool SetItemToFree(int itemIndex);
	item_t* GetItemPointer(int itemIndex);
};


template<class item_t, class index_t>
class QeueD2H_Device
{
private:
	__device__ volatile int _size;
	__device__ volatile int _pushToHere;
	__device__ index_t _arraySize;
	__device__ item_t** _items;
	__device__ QeueItemStatus* _statuses;  // pointer to the array of statuses
public:
	__device__ QeueD2H_Device(int capacity, index_t arraySize, item_t** arrays, QeueItemStatus* statuses);
	__device__ ~QeueD2H_Device();
	__device__ int GetItemForWrite();
	__device__ bool SetItemToReady(int itemIndex);
	__device__ item_t* GetItemPointer(int itemIndex);
};




