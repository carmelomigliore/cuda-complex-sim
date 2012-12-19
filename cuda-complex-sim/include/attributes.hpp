/* Copyright (C) 2012 Fabrizio Gueli
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

#ifndef ATTRIBUTES_HPP_
#define ATTRIBUTES_HPP_

#include "h_parameters.hpp"
#include "h_templates.hpp"
#include "templates.cuh"
#include <boost/random/mersenne_twister.hpp>
#include <boost/random/uniform_int.hpp>
#include <boost/random/variate_generator.hpp>

/*
 * User Defined attribute
 */
typedef struct coordinates{
	float2 c;
}coord;

/*
 * Used to generate random coordinates
 */
__host__ void generatesCoordinates(coord* t)
{
	coord temp;
	boost::mt19937 gen;
	unsigned int rseed = static_cast<unsigned int>(time(0));
	gen.seed(static_cast<unsigned int>(rseed));
	boost::uniform_int<> dist(0, 2000000);
	boost::variate_generator<boost::mt19937&, boost::uniform_int<> > die(gen, dist);

	h_initAttrArray<coord>();
	initAttrArray<coord>(t);

for(uint32_t i=0; i< h_max_nodes_number; i++)
{
	temp.c.x = die();
	temp.c.y= die();
	addAttribute<coord>(temp,i);
}


}


#endif /* ATTRIBUTES_HPP_ */
