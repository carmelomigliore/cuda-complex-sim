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

#ifndef HYGRA_CUH_
#define HYGRA_CUH_

#include "parameters.cuh"
#include "attributes.hpp"


typedef struct mn{
	uint32_t node;
	float distance;
}var;


__device__ void clustering(uint32_t degtrg,coord *attr,uint32_t this_node )
{
	uint32_t f=0;
	uint32_t q = 0;
	uint32_t neigh;
	uint32_t m_neigh;
	uint32_t count_mneigh = 0;
	uint32_t count_neigh = 0;
	var *mn_array;
	int32_t *n_array;
	int32_t *nc;
	Link* temp;
	uint32_t pippo;



				while(flagw[this_node] == true){}
				if(flagr == false)
				flagr[this_node] = true;

				atomicAdd(&(dio[this_node]),1);
		for(uint32_t j=this_node*average_links_number; j<(this_node+1)*average_links_number; j++)
		{


			if(links_targets_array[j].target != -1 && links_targets_array[j].target != -2 )
			{

				neigh = links_targets_array[j].target;
				count_neigh++;

				while(flagw[neigh] == true){}
				if(flagr == false)
				flagr[neigh] = true;

				atomicAdd(&(dio[neigh]),1);

				for(uint32_t k=neigh*average_links_number; k<(neigh+1)*average_links_number; k++)
				{
					if(links_targets_array[k].target != -1)
					{
						count_mneigh++;
					}
				}

				if(atomicAdd(&(dio[neigh]),-1)==1)
				flagr[neigh]=false;
			}
			else if(links_targets_array[j].target == -2)
			{
				temp = (Link*)links_targets_array[j+1].target;
				for(uint32_t i=0; i<supplementary_links_array_size;i++)
				{
					if(temp[i].target != -1)
					{
						neigh = temp[i].target;
						count_neigh++;
						while(flagw[neigh] == true){}
						if(flagr == false)
						flagr[neigh] = true;

						atomicAdd(&(dio[neigh]),1);

						for(uint32_t s=neigh*average_links_number; s<(neigh+1)*average_links_number; s++)
						{
							if(links_targets_array[s].target != -1)
							{
							count_mneigh++;
							}
						}

						if(atomicAdd(&(dio[neigh]),-1)==1)
						flagr[neigh]=false;
					}
				}

				break;
			}

			}

		if(atomicAdd(&(dio[this_node]),-1)==1)
			flagr[this_node]=false;


		n_array=(int32_t*)malloc(count_neigh*sizeof(int32_t));
		nc=(int32_t*)malloc(count_neigh*sizeof(int32_t));
		mn_array=(var*)malloc(count_mneigh*sizeof(var));


		while(flagw[this_node] == true){}
		if(flagr == false)
		flagr[this_node] = true;

		atomicAdd(&(dio[this_node]),1);

		for(uint32_t j=this_node*average_links_number; j<(this_node+1)*average_links_number; j++)
		{
			if(links_targets_array[j].target != -1 && links_targets_array[j].target !=-2)
			{
				neigh = links_targets_array[j].target;
				n_array[q]=neigh;
				nc[q]= neigh;
				q++;

				while(flagw[neigh] == true){}
				if(flagr == false)
				flagr[neigh] = true;

				atomicAdd(&(dio[neigh]),1);
				for(uint32_t k=neigh*average_links_number; k<(neigh+1)*average_links_number; k++)
				{
					if(links_targets_array[k].target != -1)
					{
						m_neigh = links_targets_array[k].target;
						mn_array[f].node= m_neigh;
						mn_array[f].distance = calculateDistance(attr[m_neigh].c,attr[this_node].c);
						f++;
					}


				}

				if(atomicAdd(&(dio[neigh]),-1)==1)
				flagr[neigh]=false;

			}
			else if(links_targets_array[j].target == -2)
			{
				temp = (Link*)links_targets_array[j+1].target;

				for(uint32_t i=0; i<supplementary_links_array_size;i++)
				{
					if(temp[i].target != -1)
					{
						neigh = temp[i].target;
						n_array[q]=neigh;
						nc[q]= neigh;
						q++;

						while(flagw[neigh] == true){}
						if(flagr == false)
						flagr[neigh] = true;

						atomicAdd(&(dio[neigh]),1);
						for(uint32_t s=neigh*average_links_number; s<(neigh+1)*average_links_number; s++)
						{
							if(links_targets_array[s].target != -1)
							{
								m_neigh = links_targets_array[s].target;
								mn_array[f].node= m_neigh;
								mn_array[f].distance = calculateDistance(attr[m_neigh].c,attr[this_node].c);
								f++;
							}
						}

						if(atomicAdd(&(dio[neigh]),-1)==1)
						flagr[neigh]=false;
					}
				}
				break;
			}



		 }

		if(atomicAdd(&(dio[this_node]),-1)==1)
		flagr[this_node]=false;

//if 1/2 neighbors < deg trg, then connect "this_node" with all 1/2 neighbors in mn_array (if this_node is not yet linking them)


		if(count_mneigh < degtrg)
		{

			while(flagr[this_node] == true){}

			flagw[this_node] = true;
			for(uint32_t p=0; p<count_mneigh;p++)
			{
				if(!isLinked(this_node,mn_array[p].node))
					addLink2(this_node,mn_array[p].node);
			}
			flagw[this_node]= false;

			pippo = count_mneigh;

		}
//else of the 1/2-neighbors, let the deg trg closest to "this_node"(if this_node is not yet linking them)

		else if(count_mneigh >= degtrg)
		{
			var t;
			for(uint32_t h=0; h<count_mneigh;h++)
			{
				for(uint32_t k=h+1; k<count_mneigh;k++)
				{
					if(mn_array[h].distance > mn_array[k].distance)
					{
						t =mn_array[k];
						mn_array[k]= mn_array[h];
						mn_array[h]= t;
					}
				}
			}

			while(flagr[this_node] == true){}

			flagw[this_node] = true;
			for(uint32_t j=0; j<degtrg;j++)
			{
				if(!isLinked(this_node,mn_array[j].node))
					addLink2(this_node,mn_array[j].node);
			}
			flagw[this_node] = false;

			pippo = degtrg;

		}


//Finding Critical Neighbors

		for(uint32_t r=0; r< count_neigh; r++)
		{
			for(uint32_t y=0; y<pippo; y++)
			{
				if(nc[r] == mn_array[y].node)
			    nc[r]= -1;

			}
		}

		for(uint16_t p=0; p<pippo;p++)
		{
			while(flagw[mn_array[p].node] == true){}
			if(flagr == false)
			flagr[mn_array[p].node] = true;

			atomicAdd(&(dio[mn_array[p].node]),1);
			for(uint32_t k=mn_array[p].node*average_links_number; k<(neigh+1)*average_links_number; k++)
			{
				if(links_targets_array[k].target != -1 && links_targets_array[k].target != -2 )
				{
					for(uint32_t r=0; r< count_neigh; r++)
					{
						if(nc[r]!= -1 && nc[r]== links_targets_array[k].target)
							nc[r]= -1;
					}
				}

				else if(links_targets_array[k].target == -2)
				{
					temp = (Link*)links_targets_array[k+1].target;

					for(uint32_t i=0; i<supplementary_links_array_size;i++)
					{
						if(temp[i].target != -1)
						{
							for(uint32_t s=0; s< count_neigh; s++)
							{
								if(nc[s]!= -1 && nc[s]== temp[i].target)
								nc[s]= -1;
							}
						}
					}
				}

			}

			if(atomicAdd(&(dio[mn_array[p].node]),-1)==1)
			flagr[mn_array[p].node]=false;


		}

//Reducing Critical Neighbors to Essentially Critical Neighbors
		for(uint32_t r=0; r< count_neigh; r++)
		{
			while(flagw[nc[r]] == true){}
			if(flagr == false)
			flagr[nc[r]] = true;

			atomicAdd(&(dio[nc[r]]),1);
			for(uint32_t p=0; p<count_neigh; p++)
			{
				if(nc[r] != -1 && isLinked(nc[r],nc[p]) && r != p)
				nc[r]= -1;

			}

			if(atomicAdd(&(dio[nc[r]]),-1)==1)
			flagr[nc[r]]=false;
		}

/*
 * detach "this_node" from any current neighbor (in n_array) which did not make it into the
 *  set of the closest ones (mn_array),provided it is not essentially critical (nc array)
 *
 */

		for(uint32_t p=0; p<count_neigh;p++)
		{
			for(uint32_t r=0; r< pippo; r++)
			{
				if(n_array[p] == mn_array[r].node)
				n_array[p]= -1;

			}
		}

		for(uint32_t p=0; p<count_neigh;p++)
		{
			for(uint32_t r=0; r< count_neigh; r++)
			{
				if(n_array[p]!= -1 && n_array[p] == nc[r])
				n_array[p]= -1;

			}
		}

		while(flagr[this_node] == true){}

		flagw[this_node] = true;
		for(uint32_t b=0; b<count_neigh;b++)
		{
			if(n_array[b] != -1)
			{
				removeLink(this_node,n_array[b]);
			}


		}
		flagw[this_node]= false;

		free(n_array);
		free(mn_array);
		free(nc);
		free(temp);


}



__global__ void hygra(uint32_t degtrg,coord *attr)
{
	uint32_t gtid = threadIdx.x + blockIdx.x*blockDim.x;
	while(gtid<max_nodes_number)
	{
		clustering(degtrg,attr,gtid);
		gtid+=blockDim.x*gridDim.x;
	}
}



#endif /* HYGRA_CUH_ */
