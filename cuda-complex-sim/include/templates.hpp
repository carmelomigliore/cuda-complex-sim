/* Copyright (C) 2012 Carmelo Migliore
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

#ifndef TEMPLATES_HPP_
#define TEMPLATES_HPP_


template <typename T>
__device__ inline void initArray(T initValue, T* devArray, uint32_t arrayDimension){
	uint32_t tid=threadIdx.x + blockIdx.x*blockDim.x;
	#pragma unroll
	while(tid<arrayDimension){
		devArray[tid]=initValue;
		tid+=gridDim.x*gridDim.y*gridDim.z*blockDim.x*blockDim.y*blockDim.z; //increments by the number of total threads
	}
};


/*
 * Used to copy a piece of an array from global memory INTO a tile in shared memory. The number of elements in the piece is: blockDim.x*elements_per_thread
 * Nota bene: adesso funziona, per favore di cristo non toccarla più.
 */

template <typename T>
__device__ inline void copyToTile(T* source, T* tile, int32_t start, uint16_t elements_per_thread, int16_t tile_offset){		//elements_per_thread indica quanti elementi deve copiare ciascun thread. Cosï¿½ ad esempio se ï¿½ uguale a 5 e ogni blocco ï¿½ formato da 10 thread, in totale verranno copiati nella shared memory 50 elementi
	uint16_t tid=threadIdx.x; 																		//thread index in this block
	#pragma unroll
	while(tid<blockDim.x*elements_per_thread)
	{
		tile[tid+tile_offset]=source[start+tid+blockIdx.x*blockDim.x*elements_per_thread];
		tid+=blockDim.x;
	}
}


/*
 * Copy into shared memory elements of the block before, the current block and the block after.
 * Example: the current block (blockDim.x==32, elements_per_thread==5) handles the elements from 160 to 319. Then it will copy 0-150 elements,
 * 160-319 elements and 320-479 elements into a cache in shared memory.
 * It returns a pointer to the first element handled by the current thread in the tile. On the example above
 * it would return a pointer to the 160th element.
 */

template <typename T>
__device__ inline T* copyToTileReadAhead(T* source, T* tile, int32_t start, uint16_t elements_per_thread){

	if (blockIdx.x!=0 || (start > blockDim.x*elements_per_thread)) //threads of block zero must avoid to copy elements of the previous block if they are currently treating the head of the array, because there aren't elements before source[0]!!
	{
		copyToTile <T> (source, tile, start-blockDim.x*elements_per_thread, elements_per_thread,0); //copy the elements of the block before the current block
	}
	copyToTile <T> (source, tile, start, elements_per_thread,blockDim.x*elements_per_thread); //copy the elements of the current block
	copyToTile <T> (source, tile, start+blockDim.x*elements_per_thread, elements_per_thread,2*blockDim.x*elements_per_thread); //copy the elments of the block next to the current block

	return tile+blockDim.x*elements_per_thread;
}
/*
 * Used to copy back from a tile in shared memory to an array in global memory
 */

template <typename T>
__device__ inline void copyFromTile(T* target, T* tile, int32_t start, uint16_t elements_per_thread, int16_t tile_offset){		//elements_per_thread indica quanti elementi deve copiare ciascun thread. Cosï¿½ ad esempio se ï¿½ uguale a 5 e ogni blocco ï¿½ formato da 10 thread, in totale verranno copiati nella shared memory 50 elementi
	uint16_t tid=threadIdx.x; 																		//thread index in this block
	#pragma unroll
	while(tid<blockDim.x*elements_per_thread)
	{
		target[start+tid+blockIdx.x*blockDim.x*elements_per_thread]=tile[tid+tile_offset];
		tid+=blockDim.x;
	}
}

template <typename T>
__host__ inline void copyToDevice(T* source, T* target, int32_t start, int32_t end ){
	uint16_t i = 0;

	for(i=start; i<=end;i++){
		target[i] = source[i];
	}
	}


}

#endif /* TEMPLATES_HPP_ */
