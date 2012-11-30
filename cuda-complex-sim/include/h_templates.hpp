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
 * License along with Cuda-complex-sim.  If not, see <http://www.gnu.org/licenses/>.
 */

#ifndef H_TEMPLATES_HPP_
#define H_TEMPLATES_HPP_

#include <iostream>
#include <stdint.h>
#include "h_parameters.hpp"

using namespace std;

/*
 * Template used to initialize an host Array
 */

template <typename T>
__host__ inline void h_initArray(T initValue, T* hostArray, uint32_t arrayDimension){
	uint32_t tid = 0;
	while(tid<arrayDimension){
		hostArray[tid]=initValue;
		tid++;
	}
}

/*
 * Used to allocate memory (Host) for User Attribute's Array
 */
template <typename T>
__host__ inline void h_initAttrArray(){
	h_nodes_userattr_array= malloc(h_max_nodes_number*sizeof(T));
	if(h_nodes_userattr_array == NULL){
		cerr << "\nCouldn't allocate memory on host 7";
	}
}

/*
 * Function used to add an user attribute in the Attribute's Array
 */
template <typename T>
__host__ inline void addAttribute(T attr, uint32_t node){
	((T*)h_nodes_userattr_array)[node] = attr;
}


#endif /* H_TEMPLATES_HPP_ */
