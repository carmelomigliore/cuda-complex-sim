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

__device__ void lock(uint32_t *pmutex)
{
    while(atomicCAS(pmutex, 0, 1) != 0);
}

__device__ void unlock(uint32_t *pmutex)
{
    atomicExch(pmutex, 0);
}

__device__ bool isContained(uint32_t val, int32_t* array, uint32_t size)
{
	for(uint16_t i=0; i< size; i++)
	{
		if(array[i]==val)
			return true;
	}
	return false;
}

__device__ void clustering(uint32_t degtrg,uint32_t this_node )
{
	uint8_t maxiter=2;
	for(uint8_t y=0; y<maxiter; y++)
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
	Link* temp2;
	uint32_t pippo;





				if(atomicAdd(&(dio[this_node]),1)==0)   //contatore dei lettori
				lock(&mutex_array[this_node]);



		for(uint32_t j=this_node*average_links_number; j<(this_node+1)*average_links_number; j++)
		{
			if(links_targets_array[j].target != -1 && links_targets_array[j].target != -2)
			{
				neigh = links_targets_array[j].target;
				count_neigh++;

				if(atomicAdd(&(dio[neigh]),1)==0)
				lock(&mutex_array[neigh]);


				for(uint32_t k=neigh*average_links_number; k<(neigh+1)*average_links_number; k++)
				{
					if(links_targets_array[k].target != -1 && links_targets_array[k].target != -2)
					{
						count_mneigh++;
					}
					else if(links_targets_array[k].target == -2)
					{
						temp2= (Link*)links_targets_array[k+1].target;
						for(uint32_t l=0; l<supplementary_links_array_size;l++)
						{
							if(temp2[l].target != -1)
							{
							count_mneigh++;
							}
						}
						break;
					}
				}
			if(atomicSub(&(dio[neigh]),1)==1)
				unlock(&mutex_array[neigh]);
			}
			else if(links_targets_array[j].target == -2)
			{
				temp = (Link*)(links_targets_array[j+1].target);
				for(uint32_t i=0; i<supplementary_links_array_size;i++)
				{
					if(temp[i].target != -1)
					{
						neigh = temp[i].target;
						count_neigh++;

						if(atomicAdd(&(dio[neigh]),1)==0)
					    lock(&mutex_array[neigh]);


						for(uint32_t s=neigh*average_links_number; s<(neigh+1)*average_links_number; s++)
						{
							if(links_targets_array[s].target != -1 && links_targets_array[s].target != -2 )
							{
							count_mneigh++;
							}

							else if(links_targets_array[s].target == -2)
							{
								temp2 = (Link*)(links_targets_array[s+1].target);
								for(uint32_t m=0; m<supplementary_links_array_size;m++)
								{
									if(temp2[m].target != -1)
									{
										count_mneigh++;
									}
								}
								break;
							}
						}

						if(atomicSub(&(dio[neigh]),1)==1)
						unlock(&mutex_array[neigh]);
					}
				}

				break;
			}

			}


		if(atomicSub(&(dio[this_node]),1)==1)
			unlock(&mutex_array[this_node]);


		mn_array=(var*)malloc(count_mneigh*sizeof(var));
		n_array=(int32_t*)malloc(count_neigh*sizeof(int32_t));
		nc=(int32_t*)malloc(count_neigh*sizeof(int32_t));



		if(atomicAdd(&(dio[this_node]),1)==0)
		lock(&mutex_array[this_node]);

		for(uint32_t j=this_node*average_links_number; j<(this_node+1)*average_links_number; j++)
		{
			if(links_targets_array[j].target != -1 && links_targets_array[j].target !=-2)
			{
				neigh = (uint32_t) links_targets_array[j].target;
				n_array[q]=neigh;
				nc[q]= neigh;
				q++;

				if(atomicAdd(&(dio[neigh]),1)==0)
				lock(&mutex_array[neigh]);

				for(uint32_t k=neigh*average_links_number; k<(neigh+1)*average_links_number; k++)
				{
					if(links_targets_array[k].target != -1 && links_targets_array[k].target !=-2)
					{

						m_neigh = (uint32_t)(links_targets_array[k].target);
						/*if(m_neigh != -1 && m_neigh >=0 && m_neigh<max_nodes_number)
						printf("1mezzo vicino  :%d\n",m_neigh);
						else
						*/
						mn_array[f].node= m_neigh;
						mn_array[f].distance = calculateDistance(((coord*)(nodes_userattr_array))[m_neigh].c,((coord*)(nodes_userattr_array))[this_node].c);
						f++;


					}
					else if(links_targets_array[k].target == -2)
					{
						temp2 = (Link*)(links_targets_array[k+1].target);
						for(uint32_t m=0; m<supplementary_links_array_size;m++)
						{
							if(temp2[m].target != -1)
							{
								m_neigh= temp2[m].target;
								//printf("2mezzo vicino :%d\n",m_neigh);
								mn_array[f].node= m_neigh;
								mn_array[f].distance = calculateDistance(((coord*)(nodes_userattr_array))[m_neigh].c,((coord*)(nodes_userattr_array))[this_node].c);
								f++;
							}
						}
						break;
					}

				}
				if(atomicSub(&(dio[neigh]),1)==1)
				unlock(&mutex_array[neigh]);


			}
			else if(links_targets_array[j].target == -2)
			{
				temp = (Link*)(links_targets_array[j+1].target);

				for(uint32_t i=0; i<supplementary_links_array_size;i++)
				{
					if(temp[i].target != -1)
					{
						neigh = temp[i].target;
						n_array[q]=neigh;
						nc[q]= neigh;
						q++;

						if(atomicAdd(&(dio[neigh]),1)==0)
						lock(&mutex_array[neigh]);

						for(uint32_t s=neigh*average_links_number; s<(neigh+1)*average_links_number; s++)
						{
							if(links_targets_array[s].target != -1 && links_targets_array[s].target != -2)
							{
								m_neigh = links_targets_array[s].target;
								//printf("3mezzo vicino :%d\n",m_neigh);
								mn_array[f].node= m_neigh;
								mn_array[f].distance = calculateDistance(((coord*)(nodes_userattr_array))[m_neigh].c,((coord*)(nodes_userattr_array))[this_node].c);
								f++;
							}
							else if(links_targets_array[s].target == -2)
							{
								temp2 = (Link*)(links_targets_array[s+1].target);
								for(uint32_t u=0; u<supplementary_links_array_size;u++)
								{
									if(temp2[u].target != -1)
									{
										m_neigh= temp2[u].target;
									//	printf("4mezzo vicino :%d\n",m_neigh);
										mn_array[f].node= m_neigh;
										mn_array[f].distance = calculateDistance(((coord*)(nodes_userattr_array))[m_neigh].c,((coord*)(nodes_userattr_array))[this_node].c);
										f++;
									}
								}
								break;
							}
						}

						if(atomicSub(&(dio[neigh]),1)==1)
						unlock(&mutex_array[neigh]);
					}
				}
				break;
			}



		 }

	if(atomicSub(&(dio[this_node]),1)==1)
		unlock(&mutex_array[this_node]);

//if 1/2 neighbors < deg trg, then connect "this_node" with all 1/2 neighbors in mn_array (if this_node is not yet linking them)


		if(count_mneigh < degtrg)
		{

			lock(&mutex_array[this_node]);
			for(uint32_t p=0; p<count_mneigh;p++)
			{
				if(!isLinked(this_node,mn_array[p].node))
				{
					addLink2(this_node,mn_array[p].node);
				//	printf("1 mezzo vicino aggiunto%d\n",mn_array[p].node);
				}

			}
			pippo = count_mneigh;
			unlock(&mutex_array[this_node]);



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


			lock(&mutex_array[this_node]);
			for(uint32_t j=0; j<degtrg;j++)
			{
				if(!isLinked(this_node,mn_array[j].node))
				{
					addLink2(this_node,mn_array[j].node);
			//		printf("2 mezzo vicino aggiunto%d\n",mn_array[j].node);
				}

			}
			pippo = degtrg;
			unlock(&mutex_array[this_node]);



		}

// If Neighbors array == 1/2 Neighbors array then stop

bool flages=true;

for(uint32_t a=0; a<pippo;a++)
{
	if(!isContained(mn_array[a].node, n_array, count_neigh))
	{
		flages=false;
		break;
	}
}

if(flages)
	{
		if(y>=2)
		printf("Return %d\n",y);
	return;

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

			if(atomicAdd(&(dio[mn_array[p].node]),1)==0)
			lock(&mutex_array[mn_array[p].node]);

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
					temp = (Link*)(links_targets_array[k+1].target);

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


			if(atomicSub(&(dio[mn_array[p].node]),1)==1)
			unlock(&mutex_array[mn_array[p].node]);


		}

//Reducing Critical Neighbors to Essentially Critical Neighbors
		for(uint32_t r=0; r< count_neigh; r++)
		{
			if(atomicAdd(&(dio[nc[r]]),1)==0)
			lock(&mutex_array[nc[r]]);

			for(uint32_t p=0; p<count_neigh; p++)
			{
				if(nc[r] != -1 && isLinked(nc[r],nc[p]) && r != p)
				nc[r]= -1;

			}

		if(atomicSub(&(dio[nc[r]]),1)==1)
			unlock(&mutex_array[nc[r]]);
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


		lock(&mutex_array[this_node]);
		for(uint32_t b=0; b<count_neigh;b++)
		{
			if(n_array[b] != -1)
			{
				removeLink(this_node,n_array[b]);
			}


		}
		unlock(&mutex_array[this_node]);

		free(n_array);
		free(mn_array);
		free(nc);
		if(y>=2)
		printf("Fine Hygra %d\n",y);

}
}


__global__ void hygra(uint32_t degtrg)
{
	uint32_t gtid = threadIdx.x + blockIdx.x*blockDim.x;
	while(gtid<max_nodes_number)
	{
		clustering(degtrg,gtid);
		gtid+=blockDim.x*gridDim.x;
	}
}

__global__ void stampa()
{
/*	for(uint32_t i = 0; i<200;i++)
		{
			printf("D: array dei nodi:%d X:%f y:%f\n",i,((coord*)(nodes_userattr_array))[i].c.x, ((coord*)(nodes_userattr_array))[i].c.y);
		} */

/*	for(uint32_t i = max_nodes_number-50; i<max_nodes_number;i++)
			{
				printf("D: array dei nodi:%d X:%d\n",i,nodes_array[i]);
			} */

	for(uint32_t j = 0; j<200*average_links_number;j++)
					{
					printf("D: link[%d]= %ld\n",j,links_targets_array[j].target);
					}
}



#endif /* HYGRA_CUH_ */
