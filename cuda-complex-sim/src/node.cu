/* Copyright (C) 2012 Carmelo Migliore, Fabrizio Gueli
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

#include <stdint.h>
#include <stddef.h>

#include "parameters.hpp"
#include "node.hpp"
#include "link.hpp"

__device__ bool Node::addLink(Node* trg, float distance){
		uint8_t i = 0;
		while(i<=max_links_number)
		{
			if(links_dev_array[id*max_links_number+i].target!=NULL){
				i++;
			}
			else {
				links_dev_array[id*max_links_number+i].target=trg;
				links_dev_array[id*max_links_number+i].weight=distance;
				return true;
			}
		}
		return false;
	}


