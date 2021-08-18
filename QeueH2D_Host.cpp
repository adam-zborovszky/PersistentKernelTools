#include "QeueH2D_Host.h"
#include "QeueH2D_Device.h"

#include "cuda_runtime.h"
#include "CudaCheckErrors.h"
#include "QeueItemStatus.h"
#include "QeueHeader.h"
#include "MappedMemoryArray.h"


template<class item_t> QeueH2D_Host<class item_t>::QeueH2D_Host(int size)
{
	// host side init
	_size = size;
	_pushToHere = 0;

	// host side allocations
	CudaCheckErrors(cudaHostAlloc(&_statuses, _size * sizeof(QeueItemStatus), cudaHostAllocMapped)); // allocate array of statuses (mapped mem)
	for (int i = 0; i < _size; i++)
		_statuses[i] = QeueItemStatus.Free;
	_items = (item_t**)malloc(_size * sizeof(item_t*)); // allocate array of item pointers (page-able mem)
	for (int i = 0; i < _size; i++)
		CudaCheckErrors(cudaHostAlloc(&(_items[i]), sizeof(item_t), cudaHostAllocWriteCombined | cudaHostAllocDefault)); // allocate items (pinned)	 

	// create and init the device qeue on host (preparation for copying)
	_deviceQeueOnHost = (QeueH2D_Device<item_t>*)malloc(sizeof(QeueH2D_Device<item_t>)); // allocate page-able mem
	QeueItemStatus* _statusesMappingPointer;
	CudaCheckErrors(cudaHostGetDevicePointer(&_statusesMappingPointer, _statuses, 0)); // get the status mapping pointer
	_deviceQeueOnHost->Construct(size, _statusesMappingPointer);
	
	// create the device copy of this qeue class
	CudaCheckErrors(cudaMalloc(&_deviceQeueOnDevice, sizeof(QeueH2D_Device<item_t>)); // allocate mem on device to copy this class
	CudaCheckErrors(cudaMemcpy(_deviceQeueOnDevice, _deviceQeueOnHost, sizeof(QeueH2D_Device<item_t>), cudaMemcpyHostToDevice)); // (syncronized) copy of device qeue

	cudaStreamCreate(&copyStream);
	cudaEventCreate(&copyDoneEvent);
}


template<class item_t>
QeueH2D_Host<class item_t>::~QeueH2D_Host()
{
	// free device qeue internal allocation
	_deviceQeueOnHost.Destruct();
	// free device qeue object
	cudaFree(_deviceQeueOnDevice);
	free(_deviceQeueOnHost); 
	
	// delete host side allocations
	for (int i; i < _size; i++)
		delete _items[i];
	delete _items;
	delete _statuses;

	cudaEventDestroy(copyDoneEvent);
	cudaStreamDestroy(copyStream);
}


template<class item_t>
QeueH2D_Device<item_t>* QeueH2D_Host<class item_t>::GetDeviceQeue()
{
	return _deviceQeueOnDevice;
}


template<class item_t>
int QeueH2D_Host<class item_t>::GetItemForWrite()
{
	int itemIndex = _pushToHere;
	QeueItemStatus* itemStatusPointer = _statuses.HostPointer + itemIndex;
	if (*itemStatusPointer == QeueItemStatus::Free)
	{
		*itemStatusPointer = QeueItemStatus::WriteLocked;
		if (++_pushToHere == _size) _pushToHere = 0;
		return itemIndex;
	}
	else
		return -1; // the qeue is full
}


template<class item_t>
bool QeueH2D_Host<class item_t>::SetItemToReady(int itemIndex)
{
	QeueItemStatus* itemStatusPointer = _statuses.HostPointer + itemIndex;
	if (*itemStatusPointer == QeueItemStatus::WriteLocked)
	{
		_items[itemIndex].copyToDevice()
		*itemStatusPointer = QeueItemStatus::Ready;
		return true;
	}
	else
		return false; // the status was NOT WriteLocked
}


template<class item_t>
item_t* QeueH2D_Host<class item_t>::GetItemPointer(int itemIndex)
{
	return _items[itemIndex];
}




