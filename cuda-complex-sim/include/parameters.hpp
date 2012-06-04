#ifndef PARAMETERS_HPP_
#define PARAMETERS_HPP_

#include <stdint.h>


struct Node;
struct Link;

__constant__ uint32_t max_nodes_number;
__constant__ uint8_t max_links_number;
__constant__ uint32_t active_nodes_array_size;
__constant__ uint8_t message_buffer_size;
__constant__ Node* nodes_dev_array;
__constant__ Link* links_dev_array;


//TODO: add others array pointers.

#endif
