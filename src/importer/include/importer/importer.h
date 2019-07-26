#pragma once 
#include <memory>
#include <ir/graph.h>

namespace nncase
{
namespace importer
{
    ir::graph import_tflite(xtl::span<const uint8_t> model);
}
}
