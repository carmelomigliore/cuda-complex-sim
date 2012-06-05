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

/* Global constants */

__constant__ uint32_t max_nodes_number;
__constant__ uint8_t max_links_number;
__constant__ uint32_t active_nodes_array_size;
//__constant__ uint8_t message_buffer_size;    TODO aggiungere anche il cudamemcpytosymbol

/* Nodes arrays addresses */

__constant__ bool* nodes_array;
__constant__ float* nodes_coord_x_array;
__constant__ float* nodes_coord_y_array;

/* Links arrays addresse */

__constant__ int32_t* links_targets_array;  //node's id is signed
__constant__ float* links_weights_array;


#endif /* PARAMETERS_HPP_ */
