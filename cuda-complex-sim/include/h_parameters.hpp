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

#include <vector>


/* Forward declarations */
struct n_attribute;
struct Link;
/* Global constants for host network creation */
uint32_t h_max_nodes_number;
uint8_t h_average_links_number;
uint16_t h_supplementary_links_array_size;

/* Node Arrays for host network creation */
	bool* h_nodes_array;

/* Attribute Array */
	void* h_nodes_userattr_array;
	n_attribute* h_nodes_programattr_array;

/* Link Array for host network creation */
	Link* h_links_target_array;

/* Supplementary Array address vector */
	std::vector<intptr_t> h_addr;
	std::vector<intptr_t> d_addr;

/*Host Barabasi parameters */

	uint32_t* h_links_linearized_array;


#endif /* PARAMETERS_HPP_ */
