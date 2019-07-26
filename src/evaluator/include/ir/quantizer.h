#pragma once
#include <cassert>
#include <ir/graph.h>
#include <scheduler/scheduler.h>
#include <unordered_map>

namespace nncase
{
namespace ir
{
    class quantizer
    {
    public:
        template<class TIt>
        value_range<float> get_range(TIt begin, TIt end)
        {
            auto minmax = std::minmax_element(begin, end);
            return { *minmax.first, *minmax.second };
        }

        void record(ir::output_connector &connector, value_range<float> range);
        void record(ir::output_connector &connector, xtl::span<const float> data);
        value_range<float> get(ir::output_connector &connector) const;
        quant_param_t get_quant_param(value_range<float> range, int32_t bits) const;
        fixed_mul get_fixed_mul(float value, int32_t max_bits, uint8_t max_shift, bool is_signed) const;

    private:
        std::unordered_map<ir::output_connector *, value_range<float>> quant_ranges_;
    };
}
}
