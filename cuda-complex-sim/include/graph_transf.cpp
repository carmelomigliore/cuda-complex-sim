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


#include "node.hpp"
#include "link.hpp"
#include "parameters.hpp"

using namespace std;
using namespace boost;

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

__host__ void adjlistToCompactList(adjacency_list<vecS, vecS, directedS> g){
	typedef adjacency_list<vecS, vecS, directedS> Graph;

	typedef property_map<Graph, vertex_index_t>::type IndexMap;
	    IndexMap index = get(vertex_index, g);
	    typedef graph_traits<Graph>::vertex_iterator vertex_iter;
	        std::pair<vertex_iter, vertex_iter> vp;
	        for (vp = vertices(g); vp.first != vp.second; ++vp.first){
	        	int32_t n = (int32_t) index[*vp.first];
	        	 h_addNode(n);

	        }

	        for_each(vertices(g).first, vertices(g).second,
           exercise_vertex<Graph>(g));

}







#endif /* GRAPH_TRANSF_HPP_ */
