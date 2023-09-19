#pragma once
#include "runtime_utils.h"
#include <apply.h>
#include <binary.h>
#include <cassert>
#include <concat.h>
#include <functional>
#include <gather.h>
#include <hardware_context.h>
#include <layernorm.h>
#include <matmul.h>
#include <reduce.h>
#include <softmax.h>
#include <tensor.h>
#include <thread_context.h>
#include <transpose.h>
#include <unary.h>

#define __tdma_block_sync_apply_macro(func, ctx, ...)                          \
    global_hardware_ctx.lock_block(ctx.bid());                                 \
    int visited = global_hardware_ctx.mark_block_visit(ctx.bid(), ctx.tid());  \
    func(visited, __VA_ARGS__);                                                \
    global_hardware_ctx.unlock_block(ctx.bid());                               \
    global_hardware_ctx.wait_block_sync(ctx.bid(), visited);

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
tensor<T, Loc> view(tensor<T, Loc> &src, dims_t new_dims, dims_t new_strides) {
    return tensor<T, Loc>(src.data(), new_dims, new_strides);
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
    runtime_util->rt_assert(
        dest.dimension() == src.dimension(),
        (char *)"Dest and Src dimension mismatch in __tensor_copy_sync");
    apply(gsl::make_span(src.dimension()).template as_span<const size_t>(),
          [&](gsl::span<const size_t> index) -> void {
              dest.data()[offset(dest.strides(), index)] =
                  src.data()[offset(src.strides(), index)];
          });
}

template <class T, loc_t ALoc, loc_t BLoc, loc_t CLoc>
void binary(tensor<T, ALoc> &a, tensor<T, BLoc> &b, tensor<T, CLoc> &out,
            binary_op_t op) {
    kernels::binary(
        op, a.cdata().data(), b.cdata().data(), out.data().data(),
        gsl::make_span(a.dimension()).template as_span<const size_t>(),
        gsl::make_span(a.strides()).template as_span<const size_t>(),
        gsl::make_span(b.dimension()).template as_span<const size_t>(),
        gsl::make_span(b.strides()).template as_span<const size_t>(),
        gsl::make_span(out.dimension()).template as_span<const size_t>(),
        gsl::make_span(out.strides()).template as_span<const size_t>());
}

template <class T, loc_t ALoc, loc_t BLoc>
void unary(tensor<T, ALoc> &a, tensor<T, BLoc> &out, unary_op_t op) {
    kernels::unary(
        op, a.cdata().data(), out.data().data(),
        gsl::make_span(a.strides()).template as_span<const size_t>(),
        gsl::make_span(out.dimension()).template as_span<const size_t>(),
        gsl::make_span(out.strides()).template as_span<const size_t>());
}

template <typename T, loc_t ALoc, loc_t BLoc, loc_t CLoc>
void matmul(tensor<T, ALoc> &a, tensor<T, BLoc> &b, tensor<T, CLoc> &c) {
    kernels::matmul(a.cdata().data(), b.cdata().data(), c.data().data(),
                    a.dimension(), a.strides(), b.dimension(), b.strides(),
                    c.dimension(), c.strides());
}

template <typename T, loc_t ALoc, loc_t BLoc>
void reduce(tensor<T, ALoc> &input, tensor<T, BLoc> &output, reduce_op_t op,
            T init_value, dims_t axis, bool keep_dims) {
    kernels::reduce(op, &init_value, input.cdata().data(), output.data().data(),
                    input.dimension(), axis, input.strides(), output.strides(),
                    keep_dims);
}

template <typename T, loc_t ALoc, loc_t BLoc, loc_t CLoc>
void gather(tensor<T, ALoc> &input, tensor<int64_t, BLoc> &indices,
            tensor<T, CLoc> &output, int axis) {
    kernels::gather<T, int64_t>(
        input.cdata().data(), output.data().data(), input.dimension(),
        input.strides(), output.dimension(), output.strides(),
        indices.data().data(), indices.dimension(), axis);
}

template <typename T, loc_t ALoc, loc_t BLoc>
void layernorm(tensor<T, ALoc> &input, tensor<T, loc_t::local> &sum,
               tensor<T, loc_t::local> &sum_sqr, tensor<T, BLoc> &output,
               tensor<T, loc_t::local> &gamma, tensor<T, loc_t::local> &beta,
               T eps, int32_t axis, int32_t norm_size, bool rms_norm = false) {
    assert(sum.strides() == sum_sqr.strides());
    assert(is_contiguous(sum.dimension(), sum.strides()));
    assert(gamma.strides() == beta.strides());
    assert(is_contiguous(gamma.dimension(), gamma.strides()));
    kernels::layernorm(
        input.cdata().data(), sum.data().data(), sum_sqr.data().data(),
        output.data().data(), gamma.data().data(), beta.data().data(),
        input.dimension(), input.strides(), output.strides(), sum.strides(),
        gamma.strides(), eps, axis, norm_size, rms_norm);
}

template <typename T, loc_t ALoc, loc_t BLoc>
void layernorm(tensor<T, ALoc> &input, tensor<T, loc_t::local> &sum,
               tensor<T, loc_t::local> &sum_sqr, tensor<T, BLoc> &output,
               int32_t axis, int32_t norm_size, bool rms_norm = false) {
    kernels::layernorm(input.cdata().data(), sum.data().data(),
                       sum_sqr.data().data(), output.data().data(),
                       static_cast<T *>(nullptr), static_cast<T *>(nullptr),
                       input.dimension(), input.strides(), output.strides(),
                       sum.strides(), dims_t({}), static_cast<T>(1e-5), axis,
                       norm_size, rms_norm);
}

template <typename T, loc_t ALoc>
void reduce_sum_sqr(tensor<T, ALoc> &a, tensor<T, loc_t::local> &sum,
                    tensor<T, loc_t::local> &sum_sqr) {
    kernels::reduce_sum_and_sum_sqr(a.cdata().data(), sum.data().data(),
                                    sum_sqr.data().data(), a.dimension(),
                                    a.strides(), sum.strides());
}

template <typename T, loc_t BLoc>
void concat(std::initializer_list<tensor<T, loc_t::local>*> inits,
            tensor<T, BLoc> &output, size_t axis) {
    itlib::small_vector<const gsl::byte *, 8> inputs(inits.size());
    itlib::small_vector<strides_t> in_strides(inits.size());
    auto concat_dims = dims_t(inits.size(), 1);
    for (size_t i = 0; i < inits.size(); ++i) {
        if ((*(inits.begin()+i))->dimension().size() != 0) {
            concat_dims[i] = (*(inits.begin() + i))->dimension()[axis];
        }
    }

    for (auto &in : inits) {
        inputs.push_back((const gsl::byte *)(in->data().data()));
        in_strides.push_back(in->strides());
    }

    kernels::concat(inputs, output.data().data(), output.dimension(),
                    gsl::make_span(in_strides).as_span<const strides_t>(), output.strides(), axis, concat_dims);
}

template <typename T>
void mma_visit(int visited, tensor<T, loc_t::shared> &c, tensor<T> &tmp,
               bool &load_psum) {
    if (load_psum) {
        binary<T, loc_t::local, loc_t::shared, loc_t::shared>(tmp, c, c,
                                                              binary_op_t::add);
    } else {
        if (visited == 1) {
            binary<T, loc_t::local, loc_t::shared, loc_t::shared>(
                tmp, c, c, binary_op_t::add);
        } else {
            binary<T, loc_t::local, loc_t::shared, loc_t::shared>(
                tmp, c, c, binary_op_t::add);
        }
    }
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
    matmul(a, b, tmp);

    __tdma_block_sync_apply_macro(mma_visit, ctx, c, tmp, load_psum);
}

void __tdma_block_sync_apply(std::function<void(int)> func,
                             thread_context &ctx) {
    global_hardware_ctx.lock_block(ctx.bid());
    int visited = global_hardware_ctx.mark_block_visit(ctx.bid(), ctx.tid());
    func(visited);
    global_hardware_ctx.unlock_block(ctx.bid());
    global_hardware_ctx.wait_block_sync(ctx.bid(), visited);
}

#define __tdma_all_sync_apply_macro(apply_func, broadcast_func, ctx, ...)      \
    global_hardware_ctx.lock_all();                                            \
    int visited = global_hardware_ctx.mark_all_visit(ctx.bid(), ctx.tid());    \
    apply_func(visited, ctx, __VA_ARGS__);                                     \
    global_hardware_ctx.unlock_all();                                          \
    global_hardware_ctx.wait_all_sync(visited, broadcast_func);

void __tdma_all_sync_apply(std::function<void(int)> apply_func,
                           std::function<void()> broadcast_func,
                           thread_context &ctx) {
    global_hardware_ctx.lock_all();
    int visited = global_hardware_ctx.mark_all_visit(ctx.bid(), ctx.tid());
    apply_func(visited);
    global_hardware_ctx.unlock_all();
    global_hardware_ctx.wait_all_sync(visited, broadcast_func);
}

template <class T, loc_t DestLoc>
void tdma_load_async(tensor<T, DestLoc> &dest, tensor<T, loc_t::device> &&src) {
    __tensor_copy_sync(std::forward<tensor<T, DestLoc>>(dest),
                       std::forward<tensor<T, loc_t::device>>(src));
}

template <class T, loc_t SrcLoc>
void tdma_store_async(tensor<T, SrcLoc> &src, tensor<T, loc_t::device> &&dest) {
    __tensor_copy_sync(std::forward<tensor<T, loc_t::device>>(dest),
                       std::forward<tensor<T, SrcLoc>>(src));
}

template <class T> void tdma_fill_async(tensor<T, loc_t::local> &src, T value) {
    apply(gsl::make_span(src.dimension()).template as_span<const size_t>(),
          [&](gsl::span<const size_t> index) -> void {
              src.data()[offset(src.strides(), index)] = value;
          });
}

template <class T>
void tdma_fill_async(tensor<T, loc_t::shared> &src, T value) {
    apply(gsl::make_span(src.dimension()).template as_span<const size_t>(),
          [&](gsl::span<const size_t> index) -> void {
              src.data()[offset(src.strides(), index)] = value;
          });
}

template <class T>
void tdma_broadcast_async(tensor<T, loc_t::local> &src,
                          tensor<T, loc_t::shared> &dest, thread_context &ctx) {

    global_hardware_ctx.lock_block(ctx.bid());
    __tensor_copy_sync(dest, src);
    global_hardware_ctx.unlock_block(ctx.bid());
    // int cnt = global_hardware_ctx.mark_block_visit(ctx.tid());
    // global_hardware_ctx.wait_block_sync(ctx.bid(), cnt);
}

template <class T, loc_t Src, loc_t Dest>
void tdma_load_broadcast_async([[maybe_unused]] tensor<T, Dest> &dest,
                               [[maybe_unused]] tensor<T, Src> &src,
                               [[maybe_unused]] thread_context &ctx) {
    // throw std::system_error(std::make_error_code(std::errc::not_supported));
    runtime_util->rt_assert(false, (char *)"not_supported");
}

template <class T>
void reduce_async_visit_func1(int visited, thread_context &ctx,
                              tensor<T, loc_t::local> &src,
                              [[maybe_unused]] tensor<T, loc_t::local> &dest) {
    T *gather_span = nullptr;
    auto new_dims = dims_t(src.dimension());
    new_dims.insert(new_dims.begin(), BLOCKS * CORES);
    if (visited == 1) {
        if (global_hardware_ctx.global_var != nullptr) {
            runtime_util->rt_assert(false,
                                   (char *)"the global var has been used!");
        }
        gather_span =
            (T *)runtime_util->malloc(sizeof(T) * compute_size(new_dims));
        global_hardware_ctx.global_var = gather_span;
    }
    dims_t begin(new_dims.size(), 0);
    dims_t shape(new_dims);
    begin[0] = ctx.bid() * CORES + ctx.tid();
    shape[0] = 1;
    tensor<T> gather_tensor =
        tensor<T>(gsl::make_span((T *)global_hardware_ctx.global_var,
                                 compute_size(new_dims)),
                  new_dims);
    __tensor_copy_sync<T, loc_t::local, loc_t::local>(
        std::move((gather_tensor)(begin, shape)), std::move(unsqueeze(src)));
}

template <class T>
void reduce_async_visit_func2([[maybe_unused]] int visit, thread_context &ctx,
                              tensor<T, loc_t::local> &src,
                              tensor<T, loc_t::local> &dest,
                              reduce_op_t reduce_op) {
    auto new_dims = dims_t(src.dimension());
    new_dims.insert(new_dims.begin(), BLOCKS * CORES);
    tensor<T> gather_tensor =
        tensor<T>(gsl::make_span((T *)global_hardware_ctx.global_var,
                                 compute_size(new_dims)),
                  new_dims);
    // auto new_dims = dims_t(gather_tensor.dimension());
    new_dims[0] = CORES;
    auto viewed_gather_tensor =
        tensor<T>(gather_tensor.data().subspan(
                      ctx.bid() * CORES * gather_tensor.strides()[0],
                      CORES * gather_tensor.strides()[0]),
                  new_dims, gather_tensor.strides());

    reduce(viewed_gather_tensor, dest, reduce_op, static_cast<T>(0),
           dims_t({0}), false);
}

template <class T>
void tdma_reduce_async(tensor<T, loc_t::local> &src,
                       tensor<T, loc_t::local> &dest, reduce_op_t reduce_op,
                       thread_context &ctx) {
    {__tdma_all_sync_apply_macro(reduce_async_visit_func1, ([]() -> void {}),
                                 ctx, src, dest)}

    {
        __tdma_all_sync_apply_macro(
            reduce_async_visit_func2, ([]() -> void {
                runtime_util->free(global_hardware_ctx.global_var);
                global_hardware_ctx.global_var = nullptr;
            }),
            ctx, src, dest, reduce_op)
    }
}

enum class reduce_strategy_t : uint8_t {
    all,
    by_block,
};

template <class T, loc_t ALoc>
void all_reduce_async_visit_func1(int visited, thread_context &ctx,
                                  tensor<T, ALoc> &src) {

    T *gather_span = nullptr;
    auto new_dims = dims_t(src.dimension());
    new_dims.insert(new_dims.begin(), BLOCKS * CORES);
    if (visited == 1) {
        if (global_hardware_ctx.global_var != nullptr) {
            runtime_util->rt_assert(false,
                                   (char *)"the global var has been used!");
        }
        gather_span =
            (T *)runtime_util->malloc(sizeof(T) * compute_size(new_dims));
        global_hardware_ctx.global_var = (void *)gather_span;
    }
    dims_t begin(new_dims.size(), 0);
    dims_t shape(new_dims);
    begin[0] = ctx.bid() * CORES + ctx.tid();
    shape[0] = 1;
    tensor<T> gather_tensor =
        tensor<T>(gsl::make_span((T *)global_hardware_ctx.global_var,
                                 compute_size(new_dims)),
                  new_dims);
    __tensor_copy_sync<T, loc_t::local, ALoc>(
        std::move((gather_tensor)(begin, shape)), std::move(unsqueeze(src)));
}

template <class T, loc_t ALoc, loc_t BLoc>
void all_reduce_async_visit_func2_all(int visit,
                                      [[maybe_unused]] thread_context ctx,
                                      tensor<T, ALoc> &src,
                                      tensor<T, BLoc> &dest,
                                      reduce_op_t reduce_op) {
    T *gather_span = nullptr;
    T *reduced_span = nullptr;
    auto new_dims = dims_t(src.dimension());
    new_dims.insert(new_dims.begin(), BLOCKS * CORES);
    auto reduced_shape = get_reduced_shape(new_dims, dims_t({0}), false);

    if (visit == 1) {
        gather_span = (T *)global_hardware_ctx.global_var;
        tensor<T> gather_tensor = tensor<T>(
            gsl::make_span(gather_span, compute_size(new_dims)), new_dims);
        reduced_span =
            (T *)runtime_util->malloc(sizeof(T) * compute_size(reduced_shape));
        tensor<T> reduced_tensor =
            tensor<T>(gsl::make_span(reduced_span, compute_size(reduced_shape)),
                      reduced_shape);
        reduce(gather_tensor, reduced_tensor, reduce_op, (T)0, dims_t({0}),
               false);
        runtime_util->free(gather_span);
        global_hardware_ctx.global_var = reduced_span;
    }
    tensor<T> reduced_tensor =
        tensor<T>(gsl::make_span((T *)global_hardware_ctx.global_var,
                                 compute_size(reduced_shape)),
                  reduced_shape);
    __tensor_copy_sync(std::move(dest), std::move(reduced_tensor));
}

template <class T, loc_t ALoc, loc_t BLoc>
void all_reduce_async_visit_func2_by_block([[maybe_unused]] int visit,
                                           [[maybe_unused]] thread_context ctx,
                                           tensor<T, ALoc> &src,
                                           tensor<T, BLoc> &dest,
                                           reduce_op_t reduce_op) {
    auto new_dims = dims_t(src.dimension());
    new_dims.insert(new_dims.begin(), CORES * BLOCKS);

    tensor<T> gather_tensor =
        tensor<T>(gsl::make_span((T *)global_hardware_ctx.global_var,
                                 compute_size(new_dims)),
                  new_dims);

    auto new_strides = dims_t(gather_tensor.strides());
    new_dims[0] = BLOCKS;
    new_strides[0] *= CORES;

    auto viewed_gather_tensor = view(gather_tensor, new_dims, new_strides);

    reduce(viewed_gather_tensor, dest, reduce_op, static_cast<T>(0),
           dims_t({0}), false);
}

/**
 * @brief reduce across BLOCKS * CORES
 *  1. stack the src to gather tensor, shape = [BLOCKS , CORES , ...]
 *  2. reduce gather tensor, by_block mean axis 0, all mean axis [0,1]
 *  3. broadcast gather tensor to local tensor.
 * @tparam T
 * @param src local tensor
 * @param dest local tensor
 * @param reduce_op op
 * @param strategy  all = reduce [8*4], out=[0], pre_block = reduce [8*1],
 * out=[4] .
 * @param axis axis of gather tensor, note gather tensor dims = [32,...]
 * @param ctx
 */
template <class T, loc_t ALoc, loc_t BLoc>
void tdma_all_reduce_async(tensor<T, ALoc> &src, tensor<T, BLoc> &dest,
                           reduce_op_t reduce_op, reduce_strategy_t strategy,
                           thread_context &ctx) {
    {
        __tdma_all_sync_apply_macro(all_reduce_async_visit_func1,
                                    ([]() -> void {}), ctx, src)
    }
    switch (strategy) {
    case reduce_strategy_t::all: {
        __tdma_all_sync_apply_macro(
            all_reduce_async_visit_func2_all, ([]() -> void {
                runtime_util->free(global_hardware_ctx.global_var);
                global_hardware_ctx.global_var = nullptr;
            }),
            ctx, src, dest, reduce_op);
    } break;
    case reduce_strategy_t::by_block: {
        __tdma_all_sync_apply_macro(
            all_reduce_async_visit_func2_by_block, ([]() -> void {
                runtime_util->free(global_hardware_ctx.global_var);
                global_hardware_ctx.global_var = nullptr;
            }),
            ctx, src, dest, reduce_op);
    } break;
    default:
        break;
    }
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

void tdma_all_wait(thread_context &ctx) {
    __tdma_all_sync_apply([]([[maybe_unused]] int visited) -> void {},
                          []() -> void {}, ctx);
}

void tdma_cancel() {}

void tdma_status() {}

enum class sched_strategy_t : uint8_t { pin_block_tensor, normal };

void set_sched_strategy([[maybe_unused]] sched_strategy_t sch) {}