/* Copyright (C) 2012  Fabrizio Gueli
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

#ifndef H_NODE_HPP_
#define H_NODE_HPP_

#include "h_parameters.hpp"

	/*
	 * 	Create a node and add it to the nodes array. Node creation can be done in parallel.
	 */

__host__ inline void h_addNode(int32_t id){
	h_nodes_array[id]=true;
}

#endif /* H_NODE_HPP_ */
