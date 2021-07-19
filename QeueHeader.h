#pragma once

#include "QeueItemStatus.h"

/*

Qeue flat data collection
-------------------------
Contains all data, what is required to manage a one-directional qeue
This class has implemented for QeueH2D and QeueD2H classes

*/


template<class item_t>
class QeueHeader
{
public:
	int					Size;
	volatile int		NextIndex; // points to the item to write or read next
	QeueItemStatus*	Statuses; // pointer to array of statuses (on host the real one, on device the mapping pointer)
	item_t**				ItemPointers; // pointer to array of pointers to the items (where array size = Size)
};
