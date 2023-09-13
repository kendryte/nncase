#include "thread_context.h"
using namespace shared;
void Unary_0(tensor<float, loc_t::device> &buffer_0,
             tensor<float, loc_t::device> &buffer_3) {
    thread_context ctx(bid, tid);
    tensor<float, loc_t::local> buffer_1({1, 12, 2048});
    tdma_load_async(
        buffer_1,
        buffer_0({0, 0 + (12 * 4 * bid) + (12 * 1 * tid), 0}, {1, 12, 2048}));
    tensor<float, loc_t::local> buffer_2({1, 12, 2048});
    unary(buffer_1, buffer_2, unary_op_t::asin);
    // tdma_store_async(
    //     buffer_2,
    //     buffer_3({0, 0 + (384 * 4 * bid) + (384 * 1 * tid), 0}, {1, 384,
    //     2048}), ctx);
    tdma_store_async(
        buffer_2,
        buffer_3({0, 0 + (12 * 4 * bid) + (12 * 1 * tid), 0}, {1, 12, 2048}));
}
