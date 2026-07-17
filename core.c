#include "core.h"
#include "impl.h"

struct k_context {
  size_t mem_size;
  void *mem_buffer;

  int n_objects;

  struct k_object *object_begin;
  struct k_object *object_end;
};

struct k_object {
  size_t size;
  size_t offs;

  enum object_type type;
  struct k_object *next;
};
static const size_t OBJECT_SIZE = sizeof(struct k_object);

struct k_tensor {
  int ne[TENSOR_MAX_DIMS];
  int nb[TENSOR_MAX_DIMS];

  void *data;
  enum tensor_op op;

  struct k_tensor *src[TENSOR_MAX_SRC];
};
static const size_t TENSOR_SIZE = sizeof(struct k_tensor);

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
