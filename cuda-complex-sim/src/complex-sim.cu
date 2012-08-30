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

#include "device.cuh"

int main(){

	bool* nodes_dev;
	float2* nodes_coord_dev;
	Link* links_target_dev;
	int32_t* actives_dev;
	uint32_t max_nodes=10000;
	uint8_t average_links=5;
	uint32_t active_size=1000;

	if(allocateDataStructures(&nodes_dev, &nodes_coord_dev, &links_target_dev, &actives_dev, max_nodes,average_links, active_size))
	{
		printf("\nOK\n Nodes_dev_if: %x, nodes_coord_if: %x", nodes_dev, nodes_coord_dev);
	}

	test<<<BLOCKS,THREADS_PER_BLOCK,THREADS_PER_BLOCK*sizeof(Link)>>>();
	cudaThreadExit();
}
