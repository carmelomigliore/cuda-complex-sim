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


#ifndef GRAPH_TRANSF_HPP_
#define GRAPH_TRANSF_HPP_


#include <iostream>
#include <stdint.h>
#include <stdio.h>

#include <utility>                   // for std::pair
#include <algorithm>                 // for std::for_each
#include <boost/graph/graph_traits.hpp>
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/dijkstra_shortest_paths.hpp>


#include "h_node.hpp"
#include "h_link.hpp"

using namespace std;
using namespace boost;

/*
 * Struct used to define Developer node's attributes
 */
struct n_attribute{
	n_attribute() : active(true){}
	bool active;
};

struct l_attribute{
	void* luser_defined;
};

//#ifndef DIRECTED
typedef adjacency_list<vecS, vecS, directedS,n_attribute,l_attribute> Graph;

//#elif UNDIRECTED
//typedef adjacency_list<vecS, vecS, undirectedS,n_attribute,l_attribute> Graph;

//#endif



template <class Graph>
struct print_edge {
  print_edge(Graph& g) : G(g) { }

  typedef typename boost::graph_traits<Graph>::edge_descriptor Edge;
  typedef typename boost::graph_traits<Graph>::vertex_descriptor Vertex;
  void operator()(Edge e) const
  {
    typename boost::property_map<Graph, vertex_index_t>::type
      id = get(vertex_index, G);

    Vertex src = source(e, G);
    Vertex targ = target(e, G);

    int32_t s = (int32_t) id[src];
    int32_t t = (int32_t) id[targ];
    h_addLink(s,t);

  }

  Graph& G;
};

template <class Graph>
struct exercise_vertex {
  typedef typename boost::graph_traits<Graph>::vertex_descriptor Vertex;

  exercise_vertex(Graph& _g) : g(_g) { }

  void operator()(Vertex v) const
  {
    typename boost::property_map<Graph, vertex_index_t>::type
      id = get(vertex_index, g);

    ;
    for_each(out_edges(v, g).first, out_edges(v,g).second,
             print_edge<Graph>(g));
  }

  Graph& g;
};

/*
 * Function used to Transform a "Boost-Graph" Adjacency List to a "CudaComplexSim" Compact List
 */
__host__ void adjlistToCompactList(Graph g){

	typedef property_map<Graph, vertex_index_t>::type IndexMap;
	    IndexMap index = get(vertex_index, g);
	    typedef graph_traits<Graph>::vertex_iterator vertex_iter;
	        std::pair<vertex_iter, vertex_iter> vp;
	        for (vp = vertices(g); vp.first != vp.second; ++vp.first){
	        	int32_t n = (int32_t) index[*vp.first];
	        	 h_addNode(n);
	        	 h_nodes_programattr_array[n]= g[n];

	        }

	        for_each(vertices(g).first, vertices(g).second,
           exercise_vertex<Graph>(g));
}

/*
 * Function used to Transform a "CudaComplexSim" Compact List to a "Boost-Graph" Adjacency List
 */
__host__ void CompactListToAdjList(Graph* g){

	(*g).clear();
	for(uint32_t v=0; v<h_max_nodes_number;v++)
	{
		add_vertex(*g);
		if(h_nodes_array[v] == -1)  //Se sul device è stato cancellato un nodo esso verrà settato come "non attivo"
		{
			(*g)[v].active = false;
			h_nodes_programattr_array[v].active = false;
		}
	}
	typedef std::pair<int, int> Edge;
	int t = 0;
	Link* temp;

	for(uint32_t i=0; i<h_max_nodes_number;i++)
	{
		for(uint16_t j=0; j<h_average_links_number;j++)
		{
			if(h_links_target_array[i*h_average_links_number+j].target!=-1 && h_links_target_array[i*h_average_links_number+j].target!=-2 )
			{// Se il link è presente e non è presente la lista di trabocco
				t = h_links_target_array[i*h_average_links_number+j].target;
				add_edge(i, t, *g);
			}
			else if(h_links_target_array[i*h_average_links_number+j].target==-2)
			{ //Se è presente la lista di trabocco
				temp= (Link*)h_links_target_array[i*h_average_links_number+j+1].target;
				for(uint16_t k=0; k<h_supplementary_links_array_size;k++)
				{
					if(temp[k].target != -1)
					{
						t = temp[k].target;
						add_edge(i, t, *g);
					}
				}
				break;
			}

		}
	}
}

/*
 * Function that calculates the maximum number of nodes and average links number given a Graph
 */
__host__ void calcParameters(Graph g)
{
	 int nodes = boost::num_vertices(g);
	 int edges = num_edges(g);
	 h_max_nodes_number = nodes;
	 if(edges/nodes == 0)
		h_average_links_number=2;
	 else
	 h_average_links_number= edges/nodes;

}



#endif /* GRAPH_TRANSF_HPP_ */

