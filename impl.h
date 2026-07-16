#pragma once

#include "core.h"

// bitset

typedef uint32_t bitset_t;

#define BITSET_SHR 5 // log2(sizeof(bitset_t)*8) // >>BITSET_SHR = divide by 32
#define BITSET_MASK (sizeof(bitset_t) * 8 - 1) // &BITSET_MASK = modulo 32

// how many uint32_t to store n bits
static inline size_t bitset_size(size_t n) {
  return (n + BITSET_MASK) >> BITSET_SHR;
}

// reading bit i
static inline bool bitset_get(const bitset_t *bitset, size_t i) {
  return !!(bitset[i >> BITSET_SHR] & (1u << (i & BITSET_MASK)));
}

// setting bit i
static inline void bitset_set(bitset_t *bitset, size_t i) {
  bitset[i >> BITSET_SHR] |= (1u << (i & BITSET_MASK));
}

// clear bit i
static inline void bitset_clear(bitset_t *bitset, size_t i) {
  bitset[i >> BITSET_SHR] &= ~(1u << (i & BITSET_MASK));
}

// hash set

#define HASHSET_FULL ((size_t)-1)
#define HASHSET_ALREADY_EXISTS ((size_t)-2)

struct hash_set {
  size_t size;            // size of bitset
  bitset_t *used;         // whether or not the keys are in use i.e. set
  struct k_tensor **keys; // actual tensors in the set, keys[i] is only
                          // defined if bitset_get(used, i)
};

struct k_cgraph {
  size_t size;
  int n_nodes;
  int n_leafs;

  struct k_tensor **nodes;
  struct k_tensor **leafs;
  struct k_tensor **grads;
  struct k_tensor **grad_accs;

  int32_t *use_counts;
  struct hash_set visited_hash_set;
};

// Rarely called; uses standard external linkage and relies on typical function
// call mechanisms.

struct hash_set hash_set_new(size_t size);
void hash_set_free(struct hash_set *hash_set);

// returns the minimum size for a hash set that can hold min_sz elements
size_t hash_size(size_t min_sz);

// remove all elements from the hash set
void hash_set_reset(struct hash_set *hash_set);

// Static inline for hot paths: called frequently in tight loops (e.g.,
// graph traversal, scheduling), this eliminates call overhead and
// improves optimization opportunities.

// returns true if key is in the hash set

static bool hash_contains(const struct hash_set *hash_set,
                          struct k_tensor *key);

// returns HASHSET_FULL if table is full, otherwise the current index of
// the key or where it should be inserted
static size_t hash_find(const struct hash_set *hash_set,
                        const struct k_tensor *key);

// returns HASHSET_ALREADY_EXISTS if key already exists, index otherwise,
// asserts if table is full
static size_t hash_insert(struct hash_set *hash_set, struct k_tensor *key);

// return index, asserts if table is full
static size_t hash_find_or_insert(struct hash_set *hash_set,
                                  struct k_tensor *key);

// hash function for tensor
static inline size_t hash(const struct k_tensor *p) {
  // the last 4 bits are always zero due to alignment
  return (size_t)(uintptr_t)p >> 4;
}
