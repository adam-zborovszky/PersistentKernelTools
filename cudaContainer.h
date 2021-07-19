
#pragma once




/*

global memory array class 
-------------------------
for device ONLY use!
- lifetime can span kernel multiple lauches
- allocates on device side ONLY
- avoid device side concurent access of the same elements!

*/

template<class item_t, class index_t> 
class globalMemoryArray_t
{
public:
	item_t* data_d;
	index_t capacity;
	cudaError error;
public:
	globalMemoryArray_t(index_t _capacity)
	{
		capacity = _capacity;
		if ((error = cudaMalloc(&data_d, capacity * sizeof(item_t))) != cudaSuccess) return; // allocates on the device
	}

	~globalMemoryArray_t(void)
	{
		cudaFree(data_d);
	}
};


/*

pinned (page-locked) memory array class 
---------------------------------------
for device to host, host to device async transfers 
allocates on host and device side
- allocated area is cached on GPU, multiple device side READ / WRITE is possible.
- fast, but transfer is defined explicitly 
- avoid device side concurent access of the same elements!

*/

template<class item_t, class index_t> 
class PinnedMemoryVector
{
public:
	item_t* _hostPointer;
	item_t* _devicePointer;
	index_t _capacity;
	cudaError error;
public:
	PinnedMemoryVector(index_t _capacity)
	{
		_capacity = _capacity;
		if ((error = cudaMallocHost(&_hostPointer, _capacity * sizeof(item_t))) != cudaSuccess) return; // allocates on the host 
		if ((error = cudaMalloc(&_devicePointer, _capacity * sizeof(item_t))) != cudaSuccess) return; // allocates on the device
	}

	~PinnedMemoryVector(void)
	{
		cudaFreeHost(_hostPointer);
	}

	void copyToHost(cudaStream_t _stream, cudaEvent_t _event)
	{
		error = cudaMemcpyAsync(_hostPointer, _devicePointer, _capacity * sizeof(item_t), cudaMemcpyDeviceToHost, _stream);
		cudaEventRecord(_event, _stream);
	}

	void copyToDevice(cudaStream_t _stream, cudaEvent_t _event)
	{
		error = cudaMemcpyAsync(_devicePointer, _hostPointer, _capacity * sizeof(item_t), cudaMemcpyHostToDevice, _stream);
		cudaEventRecord(_event, _stream);
	}
};



/*

mapped (zero copy) memory array class
-------------------------------------
async, always accessed with PCI-Express’s low bandwidth and high latency -> slow
useful when the device has no sufficient memory (only fragments are transferred)
- device side READ or WRITE is possible only ONCE! - sure?  zero copy is used for communication between host & device
- use for small amount of data, at high occupancy to hide PCIe latency
- coalescing is critically important
- avoid device side concurent access of the same elements!

*/

template<class item_t, class index_t> 
class MappedMemoryArray
{
public:
	item_t* data_h;
	item_t* data_d;
	index_t capacity;
	cudaError error;
public:
	MappedMemoryArray(index_t _capacity)
	{
		capacity = _capacity;
		if ((error = cudaHostAlloc(&data_h, capacity * sizeof(item_t), cudaHostAllocMapped)) != cudaSuccess) return; // allocate on host side and map on device side
		if ((error = cudaHostGetDevicePointer(&data_d, data_h, 0)) != cudaSuccess) return; // get device pointer
	}

	~MappedMemoryArray(void)
	{
		cudaFreeHost(data_h);
	}

};



/*

managed (unified) memory array class
-------------------------------------
host & device side transparently accessible array
- avoid device side concurent access of the same elements!
- avoid host-device concurent access of the same elements!

my experience is, that data migration (if necessary) happens only BEFORE and AFTER kernel run
this memory model is NOT useful for host-device communication 

*/

template<class item_t, class index_t> 
class ManagedMemoryVector
{
public:
	item_t* _pointer;
	index_t _capacity;
	cudaError error;
public:
	ManagedMemoryVector(index_t _capacity)
	{
		_capacity = _capacity;
		if ((error = cudaMallocManaged(&_pointer, _capacity * sizeof(item_t))) != cudaSuccess) return; // managed memory allocation
	}

	~ManagedMemoryVector(void)
	{
		cudaFree((void*)_pointer);
	}

};







/*

mapped (zero copy) memory list class
------------------------------------
create an mapped mem instance on host & device side
then call "construct" to allocate internal vector and initialise indexes 
before freeing call destruct to free internal allocations

provides a two end container with mapped memory item vector allocation
the item references work on both host and device side

to avoid device-device and device-host racing conditions 
a mutex locking mechanism has implemented

frequently alternating host / device  access destroys the performance

*/





template<class item_t, class index_t>
class KernelInputQueue
{
private:
	item_t* _hostData; // array of stored items
	item_t* _deviceData;
	index_t _capacity;
	index_t _length;
	index_t _indexToPopFrom;
	index_t _indexToPushTo;
	int _writeLocked; // 0 = not locked, 1 = locked (on device side atomically switched to avoid racing conditions on mutex)
	cudaError error;
public:
	__host__ void construct(index_t _capacity)
	{
		if ((error = cudaHostAlloc(&_hostData, _capacity * sizeof(item_t), cudaHostAllocMapped)) != cudaSuccess) return; // allocate on host side and map on device side
		if ((error = cudaHostGetDevicePointer(&_deviceData, _hostData, 0)) != cudaSuccess) return; // get device pointer
		_writeLocked = 0;
		_capacity = _capacity;
		_length = 0;
		_indexToPopFrom = 0;
		_indexToPushTo = 0;
	}

	__host__ void destruct()
	{
		cudaFreeHost(_hostData);
	}

	__host__ void show()
	{
		printf("[ ");
		for (index_t m = 0, i = _indexToPopFrom; m < _length; ++m, i = ((i + 1) == _capacity ? 0 : i + 1))
			printf("%d, ", _hostData[i]);
		printf("]");
	}

	__host__ index_t Length()
	{
		return _length;
	}

	__forceinline__ __device__ index_t Length()
	{
		return _length;
	}


	__host__ item_t& get_h(index_t i)
	{
		assert(i < _length);
		i += _indexToPopFrom;
		i = i > (_capacity - 1) ? i - _capacity : i;
		return _hostData[i];
	}

	__host__ __forceinline__ __device__ item_t& get_d(index_t i)
	{
		assert(i < _length);
		i += _indexToPopFrom;
		i = i > (_capacity - 1) ? i - _capacity : i;
		return _deviceData[i];
	}

	__host__ void lock_h()
	{
		while (_writeLocked); // waits until mutex == 0 (unlocked) 
		_writeLocked = 1;
	}

	__host__ void unlock_h()
	{
		_writeLocked = 0;
	}

	__forceinline__ __device__ void lock_d()
	{
		while (atomicExch(&_writeLocked, 1)); // reads  mutex to "old", changes to 1 and repeats this until "old" == 0 (unlocked)
	}

	__forceinline__ __device__ void unlock_d()
	{
		atomicExch(&_writeLocked, 0);
	}

	__host__ void push_front_h(item_t value)
	{
		lock_h();
		assert(_length + 1 <= _capacity);
		_indexToPopFrom = (_indexToPopFrom == 0) ? _capacity - 1 : _indexToPopFrom - 1;
		_hostData[_indexToPopFrom] = value;
		_length += 1;
		unlock_h();
	}

	__forceinline__ __device__ void push_front_d(item_t value)
	{
		lock_d();
		assert(_length + 1 <= _capacity);
		_indexToPopFrom = (_indexToPopFrom == 0) ? _capacity - 1 : _indexToPopFrom - 1;
		_deviceData[_indexToPopFrom] = value;
		_length += 1;
		unlock_d();
	}

	__host__ void push_back_h(item_t value)
	{
		lock_h();
		assert(_length + 1 <= _capacity);
		_hostData[_indexToPushTo] = value;
		_indexToPushTo = (_indexToPushTo == _capacity - 1) ? 0 : _indexToPushTo + 1;
		_length += 1;
		unlock_h();
	}

	__forceinline__ __device__ void push_back_d(item_t value)
	{
		lock_d();
		assert(_length + 1 <= _capacity);
		_deviceData[_indexToPushTo] = value;
		_indexToPushTo = (_indexToPushTo == _capacity - 1) ? 0 : _indexToPushTo + 1;
		_length += 1;
		unlock_d();
	}

	__host__ item_t pop_front_h()
	{
		if (_length <= 0) return 999;
		lock_h();
		item_t result = _hostData[_indexToPopFrom];
		_indexToPopFrom = (_indexToPopFrom == _capacity - 1) ? 0 : _indexToPopFrom + 1;
		_length -= 1;
		unlock_h();
		return result;
	}

	__forceinline__ __device__ item_t pop_front_d()
	{
		if (_length <= 0) return 999;
		lock_d();
		item_t result = _deviceData[_indexToPopFrom];
		_indexToPopFrom = (_indexToPopFrom == _capacity - 1) ? 0 : _indexToPopFrom + 1;
		_length -= 1;
		unlock_d();
		return result;
	}

	__host__ item_t pop_back_h()
	{
		if (_length <= 0) return 999;
		lock_h();
		_indexToPushTo = (_indexToPushTo == 0) ? _capacity - 1 : _indexToPushTo - 1;
		_length -= 1;
		unlock_h();
		return _hostData[_indexToPushTo];
	}

	__forceinline__ __device__ item_t pop_back_d()
	{
		if (_length <= 0) return 999;
		lock_d();
		_indexToPushTo = (_indexToPushTo == 0) ? _capacity - 1 : _indexToPushTo - 1;
		_length -= 1;
		unlock_d();
		return _deviceData[_indexToPushTo];
	}

};


