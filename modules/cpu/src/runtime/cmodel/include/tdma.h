#pragma once
#include <apply.h>
#include <cassert>
#include <hardware_context.h>
#include <tensor.h>
#include <thread_context.h>

template <template <class, tensor_loc_t> class TS, class DT>
void __tensor_copy_sync(TS<DT, tensor_loc_t::local> &&dest,
                        TS<DT, tensor_loc_t::local> &&src) {
    assert(dest.dimension() == src.dimension());
    apply(gsl::make_span(src.dimension()).template as_span<const size_t>(),
          [&](gsl::span<const size_t> index) -> void {
              dest.data()[offset(dest.strides(), index)] =
                  src.data()[offset(src.strides(), index)];
          });
}

template <template <class, tensor_loc_t> class TS, class DT>
void __tensor_copy_sync(TS<DT, tensor_loc_t::local> &&dest,
                        TS<DT, tensor_loc_t::device> &&src) {
    assert(dest.dimension() == src.dimension());
    apply(gsl::make_span(src.dimension()).template as_span<const size_t>(),
          [&](gsl::span<const size_t> index) -> void {
              dest.data()[offset(dest.strides(), index)] =
                  src.data()[offset(src.strides(), index)];
          });
}

template <template <class, tensor_loc_t> class TS, class DT>
void __tensor_copy_sync(TS<DT, tensor_loc_t::shared> &&dest,
                        TS<DT, tensor_loc_t::local> &&src,
                        thread_context &ctx) {
    assert(dest.dimension() == src.dimension());
    apply(gsl::make_span(src.dimension()).template as_span<const size_t>(),
          [&](gsl::span<const size_t> index) -> void {
              dest.shared_data(ctx.bid())[offset(dest.strides(), index)] =
                  src.data()[offset(src.strides(), index)];
          });
}

template <class T>
void __tensor_add_sync(tensor<T, tensor_loc_t::local> &a,
                       tensor<T, tensor_loc_t::local> &b,
                       tensor<T, tensor_loc_t::local> &out) {
    assert(a.dimension() == b.dimension());
    assert(a.dimension() == out.dimension());
    apply(a.dimension(), [&](gsl::span<size_t> index) -> void {
        out.data()[offset(out.strides(), index)] =
            (a.data()[offset(a.strides(), index)] +
             b.data()[offset(b.strides(), index)]);
    });
}

template <class T>
void tdma_load_async(tensor<T, tensor_loc_t::local> &dest,
                     tensor<T, tensor_loc_t::device> &&src) {
    __tensor_copy_sync(std::forward<tensor<T, tensor_loc_t::local>>(dest),
                       std::forward<tensor<T, tensor_loc_t::device>>(src));
}

template <class T>
void tdma_store_async(tensor<T, tensor_loc_t::local> &src,
                      tensor<T, tensor_loc_t::device> &&dest) {
    __tensor_copy_sync(dest, src);
}

template <class T>
void tdma_fill_async(tensor<T, tensor_loc_t::local> &src, T data) {
    apply(src.dimension(), [&](gsl::span<size_t> index) -> void {
        src.data()[offset(src.strides(), index)] = data;
    });
}

template <class T>
void tdma_broadcast_async(tensor<T, tensor_loc_t::local> &src,
                          tensor<T, tensor_loc_t::shared> &dest,
                          thread_context &ctx) {

    global_hardware_ctx->lock_block(ctx.bid());
    __tensor_copy_sync(dest, src);
    // int cnt = global_hardware_ctx->mark_block_visit(ctx.tid());
    global_hardware_ctx->unlock_block(ctx.bid());
    // global_hardware_ctx->wait_block_sync(ctx.bid(), cnt);
}

template <class T, tensor_loc_t Src, tensor_loc_t Dest>
void tdma_load_broadcast_async([[maybe_unused]] tensor<T, Dest> &dest,
                               [[maybe_unused]] tensor<T, Src> &src,
                               [[maybe_unused]] thread_context &ctx) {
    throw std::system_error(std::make_error_code(std::errc::not_supported));
}

void tdma_reduce_async() {
    throw std::system_error(std::make_error_code(std::errc::not_supported));
}

enum class reduce_op_t : uint8_t {
    SUM,
};

template <class T>
void tdma_all_reduce_async(tensor<T, tensor_loc_t::local> &src,
                           thread_context &ctx,
                           reduce_op_t reduce_op = reduce_op_t::SUM) {
    switch (reduce_op) {
    case reduce_op_t::SUM: {
        global_hardware_ctx->lock_all();
        int visited = global_hardware_ctx->mark_all_visit(ctx.bid(), ctx.tid());
        if (global_hardware_ctx->all_reduce_var == nullptr) {
            global_hardware_ctx->all_reduce_var =
                (void *)malloc(sizeof(tensor<T>));
            auto reduce_tensor = tensor<T>(src.dimension());
            tdma_fill_async(reduce_tensor, (T)0);
            memcpy(global_hardware_ctx->all_reduce_var, &reduce_tensor,
                   sizeof(tensor<T>));
        }
        auto reduce_tensor = (tensor<T> *)global_hardware_ctx->all_reduce_var;
        __tensor_add_sync(src, *reduce_tensor, *reduce_tensor);
        global_hardware_ctx->unlock_all();
        global_hardware_ctx->wait_all_sync(visited);
        break;
    }
    default:
        throw std::system_error(std::make_error_code(std::errc::not_supported));
        break;
    }
}

void tdma_gather_async() {}

void tdma_all_gather_async() {}

void tdma_scatter_async() {}

void tdma_wait() {}

void tdma_cancel() {}

void tdma_status() {}