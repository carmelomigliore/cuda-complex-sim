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

#ifndef MESSAGE_HPP_
#define MESSAGE_HPP_

#include "parameters.hpp"

struct __align__(16)message_t{
	int32_t sender;
	void* message;
};

__device__ void sendMessage(int32_t dest, message_t message, message_t* message_tile)
{
	uint32_t tid = threadIdx.x + blockIdx.x*blockDim.x;
	if (dest >=(blockIdx.x-1)*blockDim.x && dest < (blockIdx.x+1)*blockDim.x) //cioè se il destinatario si trova nella cache
	{
		message_tile[dest-tid]= message;
	}
	else
	{
		message_array[dest]= message;
	}
}

#endif /* MESSAGE_HPP_ */
