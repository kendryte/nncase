#include <tdma.h>

namespace shared {
static tensor<float, tensor_loc_t::shared> X({384, 8192});
static tensor<float, tensor_loc_t::shared> kh({8, 384, 128});
    // vh({8, 384, 128});                                         // [8, 384, 128]
// static tensor<float, tensor_loc_t::shared> qkh({8, 384, 384}); // [8, 384, 384]
} // namespace shared