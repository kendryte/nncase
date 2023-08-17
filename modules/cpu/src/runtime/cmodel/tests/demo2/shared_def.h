#include <tdma.h>

namespace shared {
static tensor<float, loc_t::shared> X({384, 8192});
static tensor<float, loc_t::shared> kh({8, 384, 128});
static tensor<float, loc_t::shared> vh({8, 384, 128});  // []
static tensor<float, loc_t::shared> qkh({8, 384, 384}); // []
// static tensor<float, loc_t::shared> qkh({8, 384, 384}); // []
} // namespace shared