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
	int32_t receiver;
	uint32_t ttl;
};


__device__ bool sendMessage(uint32_t receiver, message_t m)
{
	uint32_t position_offset;
	position_offset=atomicAdd(&message_counter[receiver],1);
	if(position_offset<average_links_number)
	{
		message_array[receiver+position_offset]=m;
		return true;
	}
	else
	{
		return false; //inbox full
	}
}


#endif /* MESSAGE_HPP_ */
