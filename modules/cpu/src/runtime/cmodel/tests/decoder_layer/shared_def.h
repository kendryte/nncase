#include <tdma.h>

namespace shared {
static tensor<float, loc_t::shared>
    *V2; // [1, 64, 384, 128] [1, 64, 48@b, 128]

static tensor<float, loc_t::shared> *V5; // {1, 64, 48, 128}

static tensor<float, loc_t::shared> *V13; // {1, 64, 48, 128}

static tensor<float, loc_t::shared>
    *V16; // [1, 64, 48, 128] [1, 64, 48@b, 128]

static tensor<float, loc_t::shared>
    *V25; // [1, 64, 384, 128] [1, 8@b, 384, 384]
static tensor<float, loc_t::shared>
    *V26; // [1, 64, 384, 128] [1, 64, 48@b, 384]

static tensor<float, loc_t::shared>
    *V31; // [1, 64, 384, 128] [1, 64, 48@b, 128]

static tensor<float, loc_t::shared>
    *V32; // [1, 64, 384, 128] [1, 64, 48@b, 128]
static tensor<float, loc_t::shared>
    *V33; // [1, 384, 64, 128] [1, 48@b, 64, 128]

static tensor<float, loc_t::shared> *V35; // [1, 384, 8192]

static tensor<float, loc_t::shared>
    *V38; // [1, 384, 22016] [1, 48@b, 22016]

static tensor<float, loc_t::shared>
    *V40; // [1, 384, 8192] [1, 48@b, 22016]

static tensor<float, loc_t::shared>
    *V42; // [1, 384, 8192] [1, 48@b, 8192]
} // namespace shared