#include <tdma.h>

namespace shared {
static tensor<float, loc_t::shared>
    V2({1, 8, 384, 128}); // [1, 64, 384, 128] [1, 8@b, 384, 128]
static tensor<float, loc_t::shared>
    V16({1, 8, 384, 128}); // [1, 64, 384, 128] [1, 8@b, 384, 128]
static tensor<float, loc_t::shared>
    V26({1, 8, 384, 384}); // [1, 64, 384, 128] [1, 8@b, 384, 384]

static tensor<float, loc_t::shared>
    V31({1, 8, 384, 128}); // [1, 64, 384, 128] [1, 8@b, 384, 128]

static tensor<float, loc_t::shared>
    V32({1, 8, 384, 128}); // [1, 64, 384, 128] [1, 8@b, 384, 128]
static tensor<float, loc_t::shared>
    V33({1, 384, 8, 128}); // [1, 384, 64, 128] [1, 384, 8@b, 128]

} // namespace shared