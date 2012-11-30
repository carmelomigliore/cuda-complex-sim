/* Copyright (C) 2012 Carmelo Migliore
 * 
 * This file is part of Cuda-complex-sim
 * 
 * Cuda-complex-sim is free software: you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation, either
 * version 3 of the License, or (at your option) any later version.
 *
 * Cuda-complex-sim is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with this library.  If not, see <http://www.gnu.org/licenses/>.
 */

#ifndef TASK_CUH_
#define TASK_CUH_

#include "parameters.cuh"
#include <stddef.h>

struct __align__(16) task_arguments{
	void* in;
	void* out;
};

/* Assign a task to nodes. */

__device__ void assignTask(task_t task)
{
		*task_array=task;
}


#endif /* TASK_CUH_ */
