#pragma once

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define K_GRAPH_DEFAULT_SIZE 2048
#define TENSOR_MAX_DIMS 3
#define TENSOR_MAX_SRC 10
#define K_PAD(x, n)                                                            \
  ((x + n - 1) &                                                               \
   ~(n - 1)) // round x to the next multiple of n (n is a power of 2)

// basic tensor operations

enum tensor_op {
  TENSOR_OP_NONE,
  TENSOR_OP_ADD,
  TENSOR_OP_SUB,
  TENSOR_OP_MUL,
  TENSOR_OP_DIV
};

// object type

enum object_type { OBJECT_TENSOR, OBJECT_GRAPH };

// init params

struct init_params {
  size_t mem_size;
  void *mem_buffer;
};

// define core structure

struct k_context;
struct k_object;
struct k_cgraph;
struct k_tensor;

// tensor api
struct k_tensor *new_tensor(struct k_context *ctx, int n_dims,
                            const int64_t *ne);
struct k_tensor *new_tensor_1d(struct k_context *ctx, int64_t ne0);
struct k_tensor *new_tensor_2d(struct k_context *ctx, int64_t ne0, int64_t ne1);
struct k_tensor *new_tensor_3d(struct k_context *ctx, int64_t ne0, int64_t ne1,
                               int64_t ne3);

// cgraph api
struct k_cgraph *new_graph(struct k_context *ctx, size_t size, bool grads);
void graph_build_forward(struct k_cgraph *cgraph, struct k_tensor *tensor);
size_t graph_visit_parents(struct k_cgraph *cgraph, struct k_tensor *tensor);
