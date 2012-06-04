/*
 * node_resources.hpp
 *
 *  Created on: 03/giu/2012
 *      Author: carmelo
 */

#ifndef NODE_RESOURCE_HPP_
#define NODE_RESOURCE_HPP_

struct NodeResource {
	int cpu_power;
	int ram;
	int disk_space;
	__host__ __device__ NodeResource(){
	}
	__host__ __device__ NodeResource(int cpu, int memory, int disk){
		cpu_power=cpu;
		ram=memory;
		disk_space=disk;
	}
};


#endif /* NODE_RESOURCE_HPP_ */
