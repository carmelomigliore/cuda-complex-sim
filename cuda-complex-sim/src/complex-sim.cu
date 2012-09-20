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
#include "parameters.hpp"

int main(){

	bool* nodes_dev;
	float2* nodes_coord_dev;
	Link* links_target_dev;
	task_t* task_dev;
	task_arguments* task_args_dev;
	message_t* inbox_dev;
	message_t* outbox_dev;
	int32_t* inbox_counter_dev;
	int16_t* outbox_counter_dev;
	uint32_t* barabasi_links;
	int32_t* actives_dev;
	curandState *d_state;



	if(allocateDataStructures(&nodes_dev, &nodes_coord_dev, &task_dev, &task_args_dev, &links_target_dev, &inbox_dev, &outbox_dev, &inbox_counter_dev, &outbox_counter_dev, &d_state, &barabasi_links, &actives_dev, max_nodes,average_links, active_size,supplementary_size, max_messages,barabasi_initial_nodes))
	{
		//printf("\nOK\n Nodes_dev_if: %x, nodes_coord_if: %x", nodes_dev, links_target_dev);
	}

	srand(time(NULL));
	init_stuff<<<BLOCKS,THREADS_PER_BLOCK>>>(d_state, rand());

	scale_free<<<BLOCKS,THREADS_PER_BLOCK>>>(d_state);
	size_t avail;
	size_t total;
	cudaMemGetInfo( &avail, &total );
	size_t used = total - avail;
	printf("\nMemoria: totale %d, in uso %d, disponibile: %d", total, used, avail);
	message_test<<<BLOCKS,THREADS_PER_BLOCK>>>();
	message_test2nd<<<BLOCKS,THREADS_PER_BLOCK>>>();
	message_test2nd<<<BLOCKS,THREADS_PER_BLOCK>>>();
	cudaThreadExit();
}
