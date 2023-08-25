#include <tdma.h>

namespace shared {
tensor<float, loc_t::shared> V2({1, 8, 384, 128}); // [1, 64, 384, 128] [1, 8@b, 384, 128]
} // namespace shared