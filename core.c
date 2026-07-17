#include "core.h"
#include "impl.h"
#include <assert.h>
#include <stdint.h>

// context definition of k-tensor
struct k_context {
  size_t mem_size;
  void *mem_buffer;

  int n_objects;

  struct k_object *object_begin;
  struct k_object *object_end;
};

// object definition of k-tensor
struct k_object {
  size_t size;
  size_t offs;

  enum object_type type;
  struct k_object *next;
};
static const size_t OBJECT_SIZE = sizeof(struct k_object);

// tensor definition of k-tensor
struct k_tensor {
  int64_t ne[TENSOR_MAX_DIMS];
  int nb[TENSOR_MAX_DIMS];

  void *data;
  enum tensor_op op;

  struct k_tensor *src[TENSOR_MAX_SRC];
};
static const size_t TENSOR_SIZE = sizeof(struct k_tensor);

////////////////////////////////////////////////////////////////////////////////

struct k_context *init_context(struct init_params params) {
  struct k_context *ctx = malloc(sizeof(struct k_context));
  *ctx = (struct k_context){.mem_size = params.mem_size,
                            .mem_buffer =
                                params.mem_buffer
                                    ? params.mem_buffer
                                    : malloc(params.mem_size * sizeof(char *)),
                            .n_objects = 0,
                            .object_begin = NULL,
                            .object_end = NULL};
  return ctx;
}

void free_context(struct k_context *ctx) {
  free(ctx->mem_buffer);
  free(ctx);
}

////////////////////////////////////////////////////////////////////////////////

static struct k_object *new_object(struct k_context *ctx, enum object_type type,
                                   size_t size) {
  struct k_object *obj_cur = ctx->object_end;

  size_t cur_offs = obj_cur == NULL ? 0 : obj_cur->offs;
  size_t cur_size = obj_cur == NULL ? 0 : obj_cur->size;
  size_t cur_end = cur_offs + cur_size;

  struct k_object *obj_new =
      (struct k_object *)(char *)ctx->mem_buffer + cur_end;

  *obj_new = (struct k_object){
      .size = size, .offs = cur_end + OBJECT_SIZE, .type = type, .next = NULL};

  if (obj_cur) {
    obj_cur->next = obj_new;
  } else {

    ctx->object_begin = obj_new;
  }

  ctx->object_end = obj_new;

  return obj_new;
}

////////////////////////////////////////////////////////////////////////////////

struct k_tensor *new_tensor(struct k_context *ctx, int n_dims,
                            const int64_t *ne) {

  size_t data_size = 1;

  for (int i = 0; i < n_dims; ++i) {
    data_size *= ne[i] * sizeof(float);
  }

  struct k_object *obj_new =
      new_object(ctx, OBJECT_TENSOR, TENSOR_SIZE + data_size);

  struct k_tensor *result =
      (struct k_tensor *)((char *)ctx->mem_buffer + obj_new->offs);

  *result = (struct k_tensor){.ne = {0, 0, 0},
                              .nb = {0, 0, 0},
                              .data = (void *)(result + 1),
                              .op = TENSOR_OP_NONE};
  ctx->n_objects++;

  return result;
}

struct k_tensor *new_tensor_1d(struct k_context *ctx, int64_t ne0) {
  return new_tensor(ctx, 1, &ne0);
}
struct k_tensor *new_tensor_2d(struct k_context *ctx, int64_t ne0,
                               int64_t ne1) {
  const int64_t ne[2] = {ne0, ne1};
  return new_tensor(ctx, 2, ne);
}

struct k_tensor *new_tensor_3d(struct k_context *ctx, int64_t ne0, int64_t ne1,
                               int64_t ne3) {
  const int64_t ne[3] = {ne0, ne1, ne3};
  return new_tensor(ctx, 2, ne);
}

static void *incr_ptr(void **ptr, size_t size) {
  *ptr = (void *)((char *)ptr + size);
  return ptr;
}

static size_t cgraph_nbytes(size_t size) {
  size_t size_needed = hash_size(size * 2);
  void *p = 0;
  incr_ptr(&p, sizeof(struct k_cgraph));
  incr_ptr(&p, size * sizeof(struct k_tensor *));              // nodes;
  incr_ptr(&p, size * sizeof(struct k_tensor *));              // leafs;
  incr_ptr(&p, size_needed * sizeof(int32_t));                 // used_counts;
  incr_ptr(&p, bitset_size((size_needed) * sizeof(bitset_t))); // hash_used
  return (size_t)p;
}

struct k_cgraph *new_cgraph(struct k_context *ctx, size_t size) {

  size_t obj_size = cgraph_nbytes(size);
  size_t size_needed = hash_size(size * 2);
  struct k_object *obj_new = new_object(ctx, OBJECT_GRAPH, obj_size);
  struct k_cgraph *cgraph =
      (struct k_cgraph *)((char *)ctx->mem_buffer + obj_new->offs);
  void *p = cgraph + 1;
  struct k_tensor **node_ptrs = incr_ptr(&p, size * sizeof(struct k_tensor *));
  struct k_tensor **leaf_ptrs = incr_ptr(&p, size * sizeof(struct k_tensor *));
  int32_t *use_counts_ptr = incr_ptr(&p, size_needed * sizeof(int32_t));
  struct k_tensor **hash_key_ptrs =
      incr_ptr(&p, size_needed * sizeof(struct k_tensor *));
  bitset_t *hash_used =
      incr_ptr(&p, bitset_size(size_needed) * sizeof(bitset_t));

  assert(((size_t)p - (size_t)cgraph == obj_size));

  *cgraph = (struct k_cgraph){size,
                              0,
                              0,
                              node_ptrs,
                              leaf_ptrs,
                              use_counts_ptr,
                              {size_needed, hash_used, hash_key_ptrs}

  };
  return cgraph;
}

size_t k_cgraph_visit_parents(struct k_cgraph *cgraph,
                              struct k_tensor *tensor) {

  size_t node_hash_pos = hash_find(&cgraph->visited_hash_set, tensor);

  // first time visit this tensor

  if (!bitset_get(cgraph->visited_hash_set.used, node_hash_pos)) {
    cgraph->visited_hash_set.keys[node_hash_pos] = tensor;
    bitset_set(cgraph->visited_hash_set.used, node_hash_pos);
    cgraph->use_counts[node_hash_pos] = 0;
  }

  for (int i = 0; i < TENSOR_MAX_SRC; ++i) {
    struct k_tensor *src = tensor->src[i];
    if (src) {
      size_t src_hash_pos = k_cgraph_visit_parents(cgraph, src);
      cgraph->use_counts[src_hash_pos]++;
    }
  }

  if (tensor->op == TENSOR_OP_NONE) {
    cgraph->leafs[cgraph->n_leafs] = tensor;
    cgraph->n_leafs++;
  } else {
    cgraph->nodes[cgraph->n_nodes] = tensor;
    cgraph->n_nodes++;
  }
  return node_hash_pos;
}

void k_cgraph_build(struct k_cgraph *cgraph, struct k_tensor *tensor) {
  k_cgraph_visit_parents(cgraph, tensor);
}

////////////////////////////////////////////////////////////////////////////////

struct hash_set hash_set_new(size_t size) {
  size_t size_needed = hash_size(size);
  struct hash_set result;
  result.size = size_needed;
  result.used = calloc(bitset_size(size_needed), sizeof(bitset_t));
  result.keys = malloc(size_needed * sizeof(struct k_tensor *));
  return result;
}
void hash_set_free(struct hash_set *hash_set) {
  free(hash_set->used);
  free(hash_set->keys);
}

// returns the minimum size for a hash set that can hold min_sz elements
size_t hash_size(size_t min_sz) {

  // next primes after powers of two
  static const size_t primes[] = {
      2,          3,         5,        11,        17,        37,
      67,         131,       257,      521,       1031,      2053,
      4099,       8209,      16411,    32771,     65537,     131101,
      262147,     524309,    1048583,  2097169,   4194319,   8388617,
      16777259,   33554467,  67108879, 134217757, 268435459, 536870923,
      1073741827, 2147483659};
  static const size_t n_primes = sizeof(primes) / sizeof(primes[0]);

  // find the smallest prime that is larger or equal than min_sz
  size_t l = 0;
  size_t r = n_primes;
  while (l < r) {
    size_t m = (l + r) / 2;
    if (primes[m] < min_sz) {
      l = m + 1;
    } else {
      r = m;
    }
  }
  size_t sz = l < n_primes ? primes[l] : min_sz | 1;
  return sz;
}

// remove all elements from the hash set
void hash_set_reset(struct hash_set *hash_set) {
  memset(hash_set->used, 0, bitset_size(hash_set->size) * sizeof(bitset_t));
}
