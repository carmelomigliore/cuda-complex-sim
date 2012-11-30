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
/*
#include <iostream>
#include <stdint.h>
#include <stdio.h>

#include "host.hpp"
#include "h_barabasi_game.hpp"

#include <utility>                   // for std::pair
  #include <algorithm>                 // for std::for_each
  #include <boost/graph/graph_traits.hpp>
  #include <boost/graph/adjacency_list.hpp>
 #include <boost/graph/dijkstra_shortest_paths.hpp>

 using namespace boost;

 template <class Graph>
 struct print_edges {
   print_edges(Graph& g) : G(g) { }

   typedef typename boost::graph_traits<Graph>::edge_descriptor Edge;
   typedef typename boost::graph_traits<Graph>::vertex_descriptor Vertex;
   void operator()(Edge e) const
   {
     typename boost::property_map<Graph, vertex_index_t>::type
       id = get(vertex_index, G);

     Vertex src = source(e, G);
     Vertex targ = target(e, G);

     cout << "(" << id[src] << "," << id[targ] << ") ";
   }

   Graph& G;
 };

 template <class Graph>
 struct print_index {
   print_index(Graph& g) : G(g){ }

   typedef typename boost::graph_traits<Graph>::vertex_descriptor Vertex;
   void operator()(Vertex c) const
   {
     typename boost::property_map<Graph,vertex_index_t>::type
       id = get(vertex_index, G);
     cout << id[c] << " ";
   }

   Graph& G;
 };


 template <class Graph>
 struct stampa {
   typedef typename boost::graph_traits<Graph>::vertex_descriptor Vertex;

   stampa(Graph& _g) : g(_g) { }

   void operator()(Vertex v) const
   {
     typename boost::property_map<Graph, vertex_index_t>::type
       id = get(vertex_index, g);

     cout << "vertex id: " << id[v] << endl;

     cout << "out-edges: ";
     for_each(out_edges(v, g).first, out_edges(v,g).second,
              print_edges<Graph>(g));

     cout << endl;

   }

   Graph& g;
 };



int main(){


    // writing out the edges in the graph
    typedef std::pair<int, int> Edge;


Graph g = h_barabasi_game(3,1,100);
h_average_links_number = 5;
h_max_nodes_number= 100;
//calcParameters(g);

//Allocate memory for Host Compact List (Supplementary Link array size, max nodes number, average links number)
h_allocateDataStructures(200);
printf("Numero nodi :%d\n",h_max_nodes_number);
printf("Numero average_edges :%d\n",h_average_links_number);
Link p;
p.target = -1;


//Initialize Nodes Array and Links Array
h_initArray<bool>(false,h_nodes_array,100);
h_initArray<Link>(p,h_links_target_array,h_average_links_number*100);
//Convert Boost adjacency list to Compact List
adjlistToCompactList(g);

int j = 0;
for(j=0; j<h_max_nodes_number; j++){
printf("Scorro L'array dei nodi[%d]= %d\n",j,h_nodes_array[j]);
}

for(j=0; j<h_average_links_number*100; j++){
printf("Scorro L'array dei link[%d]= %d\n",j,h_links_target_array[j].target);
}

//Convert CompactList to AdjacencyList

CompactListToAdjList(&g);

 boost::property_map<Graph, vertex_index_t>::type
    id = get(vertex_index, g);

 // cout << "vertices(g) = ";
 // boost::graph_traits<Graph>::vertex_iterator vi;
  //for (vi = vertices(g).first; vi != vertices(g).second; ++vi)
   // std::cout << id[*vi] <<  " ";
  //std::cout << std::endl;

  for_each(vertices(g).first, vertices(g).second,
           stampa<Graph>(g));



return 1;
}

*/
