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

#ifndef TASK_HPP_
#define TASK_HPP_

#include "parameters.hpp"
#include <stddef.h>

typedef struct __align__(16) task_argument_s{
	void* in;
	void* out;
}task_arguments;

/* Assign a task to a node. Return true if success, false if not */

__device__ bool assignTask(uint32_t id, task_t task, task_arguments args)
{
	if(task_array[id]==NULL)
	{
		task_array[id]=task;
		task_arguments_array[id]=args;
		return true;
	}
	return false;
}


#endif /* TASK_HPP_ */
