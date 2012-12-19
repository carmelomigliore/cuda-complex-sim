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

#ifndef PARAMETERS_CUH_
#define PARAMETERS_CUH_

#include <stdint.h>
#include "curand_kernel.h"
#include <vector>

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
__constant__ uint16_t supplementary_links_array_size;
__constant__ uint16_t message_queue_size;
__constant__ bool* flagw;
__constant__ bool* flagr;
__constant__ uint32_t* dio;

/* Nodes arrays addresses */
__constant__ bool* nodes_array;

/* Attribute Array */
__constant__ void* nodes_userattr_array;
__constant__ n_attribute* nodes_programattr_array;

/* Device Links arrays addresses */
__constant__ Link* links_targets_array;  //node's id is signed

/* Device Task arrays addresses */
__constant__ task_t* task_array;
__constant__ task_arguments* task_arguments_array;

/* Device Message array address */
__constant__ message_t* message_array;

/* Curand state address */
__constant__ curandState* cstate;

/*Device Barabasi parameters */
__constant__ uint32_t* links_linearized_array;
__constant__ uint16_t initial_nodes;



#endif /* PARAMETERS_CUH_ */
