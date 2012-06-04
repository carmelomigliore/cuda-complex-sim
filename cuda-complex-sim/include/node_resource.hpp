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


#ifndef NODE_RESOURCE_HPP_
#define NODE_RESOURCE_HPP_

struct NodeResource {
	int cpu_power;
	int ram;
	int disk_space;
	__host__ __device__ NodeResource(){
	}
	__host__ __device__ NodeResource(int cpu, int memory, int disk){
		cpu_power=cpu;
		ram=memory;
		disk_space=disk;
	}
};


#endif /* NODE_RESOURCE_HPP_ */
