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
/*
#include <iostream>
#include <stdint.h>
#include <stdio.h>

#include "node.hpp"
#include "link.hpp"
#include "parameters.hpp"
#include "host.hpp"
#include "templates.hpp"
#include "graph_transf.hpp"


  #include <utility>                   // for std::pair
  #include <algorithm>                 // for std::for_each
  #include <boost/graph/graph_traits.hpp>
  #include <boost/graph/adjacency_list.hpp>
  #include <boost/graph/dijkstra_shortest_paths.hpp>

 using namespace boost;

int host_test(){


    //Number of vertices of the Graph
    const int num_vertices = 20;


    // writing out the edges in the graph
    typedef std::pair<int, int> Edge;
    Edge edge_array[] =
    { Edge(0,1), Edge(0,2), Edge(0,3), Edge(0,4),
      Edge(1,5), Edge(1,6), Edge(1,7) };
    const int num_edges = sizeof(edge_array)/sizeof(edge_array[0]);

    // Declare a graph object
	adjacency_list<vecS, vecS, directedS> g(num_vertices);

    // add the edges to the graph object
    for (int i = 0; i < num_edges; ++i)
      add_edge(edge_array[i].first, edge_array[i].second, g);


//Allocate memory for Host Compact List (Supplementary Link array size, max nodes number, average links number)
h_allocateDataStructures(200, 30, 5);
Link p;
p.target = -1;
//Initialize Nodes Array and Links Array
h_initArray(false,h_nodes_array,30);
h_initArray(p,h_links_target_array,5*30);
//Convert Boost adiancy list to Compact List
adjlistToCompactList(g);


int j = 0;
for(j=0; j<30; j++){
printf("Scorro L'array dei nodi[%d]= %d\n",j,h_nodes_array[j]);
}

for(j=0; j<5*30; j++){
printf("Scorro L'array dei link[%d]= %d\n",j,h_links_target_array[j].target);
}

return 1;
}

*/

