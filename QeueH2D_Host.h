#pragma once

#include "QeueH2D_Device.h"

#include "cuda_runtime.h"
#include "QeueItemStatus.h"
#include "QeueHeader.h"
#include "PageAbleMemoryArray.h"
#include "MappedMemoryArray.h"
#include "PinnedMemoryArray.h"


/*

One-directional, Host -> Device data transfer queue
---------------------------------------------------
Manages an array of items, what can be written on host side and read on device side
- contains a read/write locking mechanism
- creates a device class for device side functions
- racing conditions are handled on device side only, so for multiple CPU threads use separate qeues!

*/


template<class item_t> class QeueH2D_Host
{
private:
	int _size; // maximum number of items in the qeue
	int _pushToHere; // next item index to push item into
	QeueItemStatus* _statuses; // array of item statuses
	item_t** _items; // array of item pointers
	QeueH2D_Device<item_t>* _deviceQeueOnDevice; // a pointer to the device side qeue object, as it is created and initialized by the host class
	QeueH2D_Device<item_t>* _deviceQeueOnHost; // the host side instance for init and copy purposes
public:
	QeueH2D_Host(int size);
	~QeueH2D_Host();
	QeueH2D_Device<item_t>* GetDeviceQeue();
	int GetItemForWrite();
	bool SetItemToReady(int itemIndex);
	item_t* GetItemPointer(int itemIndex);
};






