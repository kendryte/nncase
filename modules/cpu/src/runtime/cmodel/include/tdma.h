#pragma once
#include <apply.h>
#include <cassert>
#include <functional>
#include <hardware_context.h>
#include <layernorm.h>
#include <matmul.h>
#include <reduce.h>
#include <softmax.h>
#include <tensor.h>
#include <thread_context.h>
#include <transpose.h>
#define SPDLOG_ACTIVE_LEVEL SPDLOG_LEVEL_DEBUG
#include <spdlog/spdlog.h>

template <class T, loc_t Loc> tensor<T, Loc> unsqueeze(tensor<T, Loc> &src) {
    auto new_dims = dims_t(src.dimension());
    new_dims.insert(new_dims.begin(), 1);
    auto new_strides = strides_t(src.strides());
    new_strides.insert(new_strides.begin(), *new_strides.begin());
    return tensor<T, Loc>(src.data(), new_dims, new_strides);
}

template <class T, loc_t Loc>
tensor<T, Loc> view_transpose(tensor<T, Loc> &src, dims_t perm) {
    auto new_dims = dims_t();
    auto new_strides = dims_t();
    for (auto axis : perm) {
        new_dims.push_back(src.dimension()[axis]);
        new_strides.push_back(src.strides()[axis]);
    }
    return tensor<T, Loc>(src.data(), new_dims, new_strides);
}

template <class T, loc_t Loc>
tensor<T, Loc> view(tensor<T, Loc> &src, dims_t new_dims) {
    return tensor<T, Loc>(src.data(), new_dims, get_default_strides(new_dims));
}

template <class T, loc_t Loc>
void transpose(tensor<T, Loc> &src, tensor<T, Loc> &dest, dims_t perm) {
    return kernels::transpose(src.cdata().data(), dest.data().data(),
                              src.dimension(), perm, src.strides(),
                              dest.strides());
}

template <loc_t SrcLoc, loc_t DestLoc>
void softmax(tensor<float, SrcLoc> &src, tensor<float, DestLoc> &dest,
             int axis) {
    kernels::softmax(src.cdata().data(), dest.data().data(), src.dimension(),
                     src.strides(), dest.strides(), axis, 1.0f, false);
}

template <class T, loc_t DestLoc, loc_t SrcLoc>
void __tensor_copy_sync(tensor<T, DestLoc> &&dest, tensor<T, SrcLoc> &&src) {
    assert(dest.dimension() == src.dimension());
    apply(gsl::make_span(src.dimension()).template as_span<const size_t>(),
          [&](gsl::span<const size_t> index) -> void {
              dest.data()[offset(dest.strides(), index)] =
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

template <typename T, loc_t ALoc, loc_t BLoc>
void tensor_reduce_sync(tensor<T, ALoc> &input, tensor<T, BLoc> &output,
                        reduce_op_t op, T init_value, dims_t axis,
                        bool keep_dims) {
    kernels::reduce(op, &init_value, input.cdata().data(), output.data().data(),
                    input.dimension(), axis, input.strides(), output.strides(),
                    keep_dims);
}

template <typename T, loc_t Loc>
void tensor_layernorm_sync(tensor<T, Loc> &input, tensor<T, loc_t::local> &sum,
                           tensor<T, loc_t::local> &sum_sqr,
                           tensor<T, loc_t::local> &gamma,
                           tensor<T, loc_t::local> &beta, T eps, int32_t axis,
                           int32_t norm_size) {
    kernels::layernorm(input.data().data(), sum.data().data(),
                       sum_sqr.data().data(), gamma.data().data(),
                       beta.data().data(), input.dimension(), input.strides(),
                       eps, axis, norm_size);
}

template <typename T, loc_t Loc>
void tensor_layernorm_sync(tensor<T, Loc> &input, tensor<T, loc_t::local> &sum,
                           tensor<T, loc_t::local> &sum_sqr, int32_t axis,
                           int32_t norm_size) {
    kernels::layernorm(input.data().data(), sum.data().data(),
                       sum_sqr.data().data(), static_cast<T *>(nullptr),
                       static_cast<T *>(nullptr), input.dimension(),
                       input.strides(), static_cast<T>(1e-5), axis, norm_size);
}

template <typename T, loc_t ALoc>
void tensor_reduce_sum_sqr(tensor<T, ALoc> &a, tensor<T, loc_t::local> &sum,
                           tensor<T, loc_t::local> &sum_sqr) {
    kernels::reduce_sum_and_sum_sqr(a.cdata().data(), sum.data().data(),
                                    sum_sqr.data().data(), a.dimension(),
                                    a.strides(), sum.strides());
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
                __tensor_binary_sync<T, loc_t::local, loc_t::shared,
                                     loc_t::shared>(
                    tmp, c, c, [](T _a, T _b) -> T { return _a + _b; });
            } else {
                if (visited == 1) {
                    __tensor_binary_sync<T, loc_t::local, loc_t::shared,
                                         loc_t::shared>(
                        tmp, c, c,
                        [](T _a, [[maybe_unused]] T _b) -> T { return _a; });
                } else {
                    __tensor_binary_sync<T, loc_t::local, loc_t::shared,
                                         loc_t::shared>(
                        tmp, c, c, [](T _a, T _b) -> T { return _a + _b; });
                }
            }
        },
        ctx);
}

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

void __tdma_all_sync_apply(std::function<void(int)> apply_func,
                           std::function<void()> broadcast_func,
                           thread_context &ctx) {
    global_hardware_ctx->lock_all();
    int visited = global_hardware_ctx->mark_all_visit(ctx.bid(), ctx.tid());
    SPDLOG_DEBUG("__tdma_all_sync_apply bid {} tid {} visited {}\n", ctx.bid(),
                 ctx.tid(), visited);
    apply_func(visited);
    global_hardware_ctx->unlock_all();
    global_hardware_ctx->wait_all_sync(visited, broadcast_func);
}

template <class T, loc_t DestLoc>
void tdma_load_async(tensor<T, DestLoc> &dest, tensor<T, loc_t::device> &&src,
                     [[maybe_unused]] thread_context &ctx) {
    __tensor_copy_sync(std::forward<tensor<T, DestLoc>>(dest),
                       std::forward<tensor<T, loc_t::device>>(src));
}

template <class T, loc_t SrcLoc>
void tdma_store_async(tensor<T, SrcLoc> &src, tensor<T, loc_t::device> &&dest,
                      [[maybe_unused]] thread_context &ctx) {
    __tensor_copy_sync(std::forward<tensor<T, loc_t::device>>(dest),
                       std::forward<tensor<T, SrcLoc>>(src));
}

template <class T> void tdma_fill_async(tensor<T, loc_t::local> &src, T value) {
    __tensor_unary_sync<T, loc_t::local, loc_t::local>(
        src, src, [&value]([[maybe_unused]] T a) -> T { return value; });
}

template <class T>
void tdma_broadcast_async(tensor<T, loc_t::local> &src,
                          tensor<T, loc_t::shared> &dest, thread_context &ctx) {

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

/**
 * @brief reduce across BLOCKS * CORES
 *  1. stack the src to gather tensor
 *  2. reduce gather tensor
 *  3. broadcast gather tensor to local tensor.
 * @tparam T
 * @param src local tensor
 * @param dest local tensor
 * @param reduce_op op
 * @param axis axis of gather tensor, note gather tensor dims = [32,...]
 * @param ctx
 */
template <class T>
void tdma_all_reduce_async(tensor<T, loc_t::local> &src,
                           tensor<T, loc_t::local> &dest, reduce_op_t reduce_op,
                           dims_t axis, thread_context &ctx) {
    __tdma_all_sync_apply(
        [&src, &ctx](int visited) -> void {
            tensor<T> *gather_tensor = nullptr;
            auto new_dims = dims_t(src.dimension());
            new_dims.insert(new_dims.begin(), 32);
            if (visited == 1) {
                if (global_hardware_ctx->global_var != nullptr) {
                    throw std::runtime_error(" the global var has been used!");
                }
                gather_tensor = new tensor<T>(new_dims);
                global_hardware_ctx->global_var = (void *)gather_tensor;
            } else {
                gather_tensor =
                    static_cast<tensor<T> *>(global_hardware_ctx->global_var);
            }
            dims_t begin(new_dims.size(), 0);
            dims_t shape(new_dims);
            begin[0] = ctx.bid() * CORES + ctx.tid();
            shape[0] = 1;
            __tensor_copy_sync<T, loc_t::local, loc_t::local>(
                std::move((*gather_tensor)(begin, shape)),
                std::move(unsqueeze(src)));
        },
        []() -> void {}, ctx);

    __tdma_all_sync_apply(
        [&]([[maybe_unused]] int visit) -> void {
            tensor<T> *gather_tensor = nullptr;
            tensor<T> *reduced_tensor = nullptr;
            if (visit == 1) {
                gather_tensor =
                    static_cast<tensor<T> *>(global_hardware_ctx->global_var);
                auto reduced_shape =
                    get_reduced_shape(gather_tensor->dimension(), axis, false);
                reduced_tensor = new tensor<T>(reduced_shape);
                tensor_reduce_sync(*gather_tensor, *reduced_tensor, reduce_op,
                                   (T)0, axis, false);
                delete gather_tensor;
                global_hardware_ctx->global_var = reduced_tensor;
            } else {
                reduced_tensor =
                    static_cast<tensor<T> *>(global_hardware_ctx->global_var);
                __tensor_copy_sync(std::move(dest), std::move(*reduced_tensor));
            }
        },
        []() -> void {
            auto reduced_tensor =
                static_cast<tensor<T> *>(global_hardware_ctx->global_var);
            delete reduced_tensor;
            global_hardware_ctx->global_var = nullptr;
        },
        ctx);
}

void tdma_gather_async() {}

void tdma_all_gather_async() {}

void tdma_scatter_async() {}

/**
 * @brief tdma inner block wait.
 *
 * @param ctx
 */
void tdma_wait(thread_context &ctx) {
    __tdma_block_sync_apply([]([[maybe_unused]] int visited) -> void {}, ctx);
}

void tdma_cancel() {}

void tdma_status() {}

enum class sched_strategy_t : uint8_t { pin_block_tensor, normal };

void set_sched_strategy([[maybe_unused]] sched_strategy_t sch) {}