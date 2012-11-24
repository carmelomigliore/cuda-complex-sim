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
/*
#include "device.cuh"
#include "parameters.hpp"

int main(int argc, char** argv){

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

	if (argc!=3)
	{
		perror("\nErrore");
		exit(1);
	}
	uint32_t max_nodes=atoi(argv[1]);
	uint8_t average_links=atoi(argv[2]);
	uint16_t max_messages=20;		//not needed in our simulation
	uint32_t active_size=1000;		//not needed in our simulation
	uint16_t supplementary_size=30; //not needed in our simulation
	uint16_t barabasi_initial_nodes=atoi(argv[2])+1;

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

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	// Start record
	cudaEventRecord(start, 0);

	message_test<<<BLOCKS,THREADS_PER_BLOCK,average_links*THREADS_PER_BLOCK*sizeof(Link)>>>();
	message_test2nd<<<BLOCKS,THREADS_PER_BLOCK,average_links*THREADS_PER_BLOCK*sizeof(Link)>>>();
	message_test2nd<<<BLOCKS,THREADS_PER_BLOCK,average_links*THREADS_PER_BLOCK*sizeof(Link)>>>();
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	float elapsedTime;
	cudaEventElapsedTime(&elapsedTime, start, stop); // that's our time!
	// Clean up:
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	FILE *file;
	file=fopen("times.txt","a");
	fprintf(file, "%f\n",elapsedTime);
	fflush(file);
	fclose(file);
	cudaThreadExit();
}

*/


