#pragma once
#include <apply.h>
#include <cassert>
#include <functional>
#include <hardware_context.h>
#include <matmul.h>
#include <tensor.h>
#include <thread_context.h>
#define SPDLOG_ACTIVE_LEVEL SPDLOG_LEVEL_DEBUG
#include <spdlog/spdlog.h>


template <template <class, loc_t> class TS, class DT>
void __tensor_copy_sync(TS<DT, loc_t::local> &&dest,
                        TS<DT, loc_t::local> &&src,
                        [[maybe_unused]] thread_context &ctx) {
    assert(dest.dimension() == src.dimension());
    apply(gsl::make_span(src.dimension()).template as_span<const size_t>(),
          [&](gsl::span<const size_t> index) -> void {
              dest.data()[offset(dest.strides(), index)] =
                  src.data()[offset(src.strides(), index)];
          });
}

template <template <class, loc_t> class TS, class DT>
void __tensor_copy_sync(TS<DT, loc_t::local> &&dest,
                        TS<DT, loc_t::device> &&src,
                        [[maybe_unused]] thread_context &ctx) {
    assert(dest.dimension() == src.dimension());
    apply(gsl::make_span(src.dimension()).template as_span<const size_t>(),
          [&](gsl::span<const size_t> index) -> void {
              dest.data()[offset(dest.strides(), index)] =
                  src.data()[offset(src.strides(), index)];
          });
}

template <template <class, loc_t> class TS, class DT>
void __tensor_copy_sync(TS<DT, loc_t::shared> &&dest,
                        TS<DT, loc_t::device> &&src,
                        [[maybe_unused]] thread_context &ctx) {

    __tdma_block_sync_apply(
        [&dest, &src](int visited) -> void {
            if (visited == 1) {
                assert(dest.dimension() == src.dimension());
                apply(gsl::make_span(src.dimension())
                          .template as_span<const size_t>(),
                      [&](gsl::span<const size_t> index) -> void {
                          dest.data()[offset(dest.strides(), index)] =
                              src.data()[offset(src.strides(), index)];
                      });
            }
        },
        ctx);
}

template <template <class, loc_t> class TS, class DT>
void __tensor_copy_sync(TS<DT, loc_t::device> &&dest,
                        TS<DT, loc_t::shared> &&src,
                        [[maybe_unused]] thread_context &ctx) {

    __tdma_block_sync_apply(
        [&dest, &src](int visited) -> void {
            if (visited == 1) {
                assert(dest.dimension() == src.dimension());
                apply(gsl::make_span(src.dimension())
                          .template as_span<const size_t>(),
                      [&](gsl::span<const size_t> index) -> void {
                          dest.data()[offset(dest.strides(), index)] =
                              src.data()[offset(src.strides(), index)];
                      });
            }
        },
        ctx);
}

template <template <class, loc_t> class TS, class DT>
void __tensor_copy_sync(TS<DT, loc_t::shared> &&dest,
                        TS<DT, loc_t::local> &&src,
                        [[maybe_unused]] thread_context &ctx) {
    assert(dest.dimension() == src.dimension());
    apply(gsl::make_span(src.dimension()).template as_span<const size_t>(),
          [&](gsl::span<const size_t> index) -> void {
              dest.data(ctx.bid())[offset(dest.strides(), index)] =
                  src.data()[offset(src.strides(), index)];
          });
}

template <class T, loc_t ALoc, loc_t BLoc, loc_t CLoc>
void __tensor_binary_sync(tensor<T, ALoc> &a, tensor<T, BLoc> &b,
                          tensor<T, CLoc> &out,
                          std::function<T(T a, T b)> callable) {
    assert(a.dimension() == b.dimension());
    assert(a.dimension() == out.dimension());
    apply(gsl::make_span(a.dimension()).template as_span<const size_t>(),
          [&](gsl::span<const size_t> index) -> void {
              out.data()[offset(out.strides(), index)] =
                  callable(a.data()[offset(a.strides(), index)],
                           b.data()[offset(b.strides(), index)]);
          });
}

template <class T, loc_t ALoc, loc_t BLoc, loc_t CLoc>
void __tensor_unary_sync(tensor<T, ALoc> &a, tensor<T, CLoc> &out,
                         std::function<T(T a)> callable) {
    assert(a.dimension() == out.dimension());
    apply(gsl::make_span(a.dimension()).template as_span<const size_t>(),
          [&](gsl::span<const size_t> index) -> void {
              out.data()[offset(out.strides(), index)] =
                  callable(a.data()[offset(a.strides(), index)]);
          });
}

template <typename T, loc_t ALoc, loc_t BLoc>
void tensor_mma_sync(tensor<T, ALoc> &a, tensor<T, BLoc> &b,
                     tensor<T, loc_t::local> &c) {
    matmul(a.cdata().data(), b.cdata().data(), c.data().data(), a.dimension(),
           a.strides(), b.dimension(), b.strides(), c.dimension(), c.strides());
}

/**
 * @brief Block 级别的 mma，每个 thread 执行自己的 mma，最终 psum 累加写入
 * shared memory
 *
 * @tparam T type
 * @tparam ALoc shared or local
 * @tparam BLoc shared or local
 * @param a
 * @param b
 * @param c must be shared
 * @param load_psum
 * @param ctx
 */
template <typename T, loc_t ALoc, loc_t BLoc>
void tensor_block_mma_sync(tensor<T, ALoc> &a, tensor<T, BLoc> &b,
                           tensor<T, loc_t::shared> &c, bool load_psum,
                           thread_context &ctx) {
    tensor<T> tmp(c.dimension());
    tensor_mma_sync(a, b, tmp);

    __tdma_block_sync_apply(
        [&](int visited) -> void {
            if (load_psum) {
                __tensor_binary_sync<T, loc_t::local,
                                     loc_t::shared,
                                     loc_t::shared>(
                    tmp, c, c, [](T _a, T _b) -> T { return _a + _b; });
            } else {
                if (visited == 1) {
                    __tensor_binary_sync<T, loc_t::local,
                                         loc_t::shared,
                                         loc_t::shared>(
                        tmp, c, c,
                        [](T _a, [[maybe_unused]] T _b) -> T { return _a; });
                } else {
                    __tensor_binary_sync<T, loc_t::local,
                                         loc_t::shared,
                                         loc_t::shared>(
                        tmp, c, c, [](T _a, T _b) -> T { return _a + _b; });
                }
            }
        },
        ctx);
}

template <class T, loc_t DestLoc>
void tdma_load_async(tensor<T, DestLoc> &dest,
                     tensor<T, loc_t::device> &&src,
                     thread_context &ctx) {
    __tensor_copy_sync(std::forward<tensor<T, DestLoc>>(dest),
                       std::forward<tensor<T, loc_t::device>>(src), ctx);
}

template <class T, loc_t SrcLoc>
void tdma_store_async(tensor<T, SrcLoc> &src,
                      tensor<T, loc_t::device> &&dest,
                      thread_context &ctx) {
    __tensor_copy_sync(std::forward<tensor<T, loc_t::device>>(dest),
                       std::forward<tensor<T, SrcLoc>>(src), ctx);
}

template <class T>
void tdma_fill_async(tensor<T, loc_t::local> &src, T data) {
    apply(src.dimension(), [&](gsl::span<size_t> index) -> void {
        src.data()[offset(src.strides(), index)] = data;
    });
}

template <class T>
void tdma_broadcast_async(tensor<T, loc_t::local> &src,
                          tensor<T, loc_t::shared> &dest,
                          thread_context &ctx) {

    global_hardware_ctx->lock_block(ctx.bid());
    __tensor_copy_sync(dest, src);
    global_hardware_ctx->unlock_block(ctx.bid());
    // int cnt = global_hardware_ctx->mark_block_visit(ctx.tid());
    // global_hardware_ctx->wait_block_sync(ctx.bid(), cnt);
}

template <class T, loc_t Src, loc_t Dest>
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
void tdma_all_reduce_async(tensor<T, loc_t::local> &src,
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

void __tdma_block_sync_apply(std::function<void(int)> func,
                             thread_context &ctx) {
    global_hardware_ctx->lock_block(ctx.bid());
    int visited = global_hardware_ctx->mark_block_visit(ctx.bid(), ctx.tid());
    SPDLOG_DEBUG("__tdma_block_sync_apply bid {} tid {} visited {}\n",
                 ctx.bid(), ctx.tid(), visited);
    func(visited);
    global_hardware_ctx->unlock_block(ctx.bid());
    global_hardware_ctx->wait_block_sync(ctx.bid(), visited);
}