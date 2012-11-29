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

#ifndef PARAMETERS_HPP_
#define PARAMETERS_HPP_

#include <stdint.h>
#include "curand_kernel.h"

#define BLOCKS 256

#define THREADS_PER_BLOCK 64



/* Forward declarations */
struct n_attribute;
struct Link;
struct task_arguments;
struct message_t;
typedef bool (*task_t) (void* in, void **out); //generic task

/* Global constants */
__constant__ uint32_t max_nodes_number;
__constant__ uint8_t average_links_number;
__constant__ uint32_t active_nodes_array_size;
__constant__ uint16_t supplementary_links_array_size;
__constant__ uint16_t message_queue_size;

/* Global constants for host network creation */
uint32_t h_max_nodes_number;
uint8_t h_average_links_number;
uint32_t h_active_nodes_array_size;
uint16_t h_supplementary_links_array_size;

/* Nodes arrays addresses */
__constant__ bool* nodes_array;
__constant__ float2* nodes_coord_array;

/* Node Arrays for host network creation */
	bool* h_nodes_array;
	float2* h_nodes_coord_array;

/* Attribute Array */
	void* h_nodes_userattr_array;
	n_attribute* h_nodes_programattr_array;
	__constant__ void* nodes_userattr_array;
	__constant__ n_attribute* nodes_programattr_array;

/* Device Links arrays addresses */
__constant__ Link* links_targets_array;  //node's id is signed

/* Link Array for host network creation */
	Link* h_links_target_array;

/* Device Task arrays addresses */
__constant__ task_t* task_array;
__constant__ task_arguments* task_arguments_array;

/* Device Message array address */
__constant__ message_t* message_array;
__constant__ message_t* outbox_array;
__constant__ int32_t* message_counter;
__constant__ int16_t* outbox_counter;

/* Message Array for host network creation */
	message_t* h_message_array;

/* Curand state address */
__constant__ curandState* cstate;

/*Device Barabasi parameters */
__constant__ uint32_t* links_linearized_array;
__constant__ uint16_t initial_nodes;

__constant__ uint32_t* fail_count;

/*Host Barabasi parameters */

uint32_t* h_links_linearized_array;
uint16_t h_initial_nodes;
uint32_t* h_fail_count;

#endif /* PARAMETERS_HPP_ */
