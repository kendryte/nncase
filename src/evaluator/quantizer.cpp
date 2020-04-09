/* Copyright 2019-2020 Canaan Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include <chrono>
#include <hlir/ops/constant.h>
#include <hlir/quantizer.h>
#include <hlir/visitor.h>

using namespace nncase;
using namespace nncase::hlir;
using namespace nncase::scheduler;

#define KLD_METHOD 0

namespace
{
value_range<float> union_range(const value_range<float> &lhs, const value_range<float> &rhs)
{
    return { std::min(lhs.min, rhs.min), std::max(lhs.max, rhs.max) };
}

value_range<float> ema_range(const value_range<float> &lhs, const value_range<float> &rhs)
{
    const float alpha = 0.01f;
    return { (1 - alpha) * lhs.min + alpha * rhs.min, (1 - alpha) * lhs.max + alpha * rhs.max };
}

float compute_kld(xtl::span<float> p, xtl::span<float> q)
{
    assert(p.size() == q.size());
    float d = 0.f;
    for (size_t i = 0; i < p.size(); i++)
    {
        if (p[i])
            d += q[i] ? p[i] * std::log(p[i] / q[i]) : 1.f;
    }

    return d;
}

float compute_l2(xtl::span<float> p, value_range<float> p_range, value_range<float> q_range, size_t q_bins)
{
    auto p_interval = (p_range.max - p_range.min) / p.size();
    auto q_interval = (q_range.max - q_range.min) / (q_bins - 1);
    float d = 0.f;

    for (size_t i = 0; i < p.size(); i++)
    {
        auto p_val = p_range.min + p_interval * (i + 0.0f);
        auto q_idx = std::clamp((int32_t)std::round((p_val - q_range.min) / q_interval), 0, (int32_t)q_bins - 1);
        auto q_val = q_range.min + q_interval * (q_idx + 0.0f);
        d += std::pow(p_val - q_val, 2) * p[i];
    }

    return d;
}
}

quantizer::quantizer(calibrate_method cali_method, size_t bins)
    : cali_method_(cali_method), bins_(bins)
{
}

void quantizer::record(hlir::output_connector &connector, value_range<float> range)
{
    auto it = quant_ranges_.find(&connector);
    if (it == quant_ranges_.end())
        quant_ranges_.emplace(&connector, range);
    else
        it->second = union_range(it->second, range);
}

void quantizer::set(hlir::output_connector &connector, value_range<float> range)
{
    quant_ranges_[&connector] = range;
}

void quantizer::record(output_connector &connector, xtl::span<const float> data)
{
    switch (stage_)
    {
    case quantize_stage::collect_range:
        record(connector, get_range(data.begin(), data.end()));
        break;
    case quantize_stage::collect_distribution:
        histograms_.at(&connector).record(data);
        break;
    default:
        throw std::runtime_error("Invalid operation in current quantization stage");
    }
}

void quantizer::begin_collect_distribution()
{
    for (auto &&p : quant_ranges_)
        histograms_.emplace(p.first, histogram(fixup_range(p.second), bins_, 256));

    stage_ = quantize_stage::collect_distribution;
}

void quantizer::end_collect_distribution(std::function<void(size_t)> progress)
{
    size_t i = 0;
    for (auto &&h : histograms_)
    {
        h.second.finish();
        quant_ranges_.at(h.first) = h.second.optimal_range();
        progress(i++);
    }
}

quant_param_t quantizer::get_quant_param(value_range<float> range, int32_t bits)
{
    range = fixup_range(range);
    auto r = range.max - range.min;
    auto scale = ((1LL << bits) - 1) / r;
    auto bias = std::round(-range.min * scale);
    assert(bias >= 0);
    return { static_cast<int32_t>(bias), scale };
}

value_range<float> quantizer::get(hlir::output_connector &connector) const
{
    return quant_ranges_.at(&connector);
}

fixed_mul quantizer::get_fixed_mul(float value, int32_t max_bits, uint8_t max_shift, bool is_signed)
{
    assert(!is_signed || value >= 0);

    auto bits = is_signed ? max_bits - 1 : max_bits;
    int32_t shift = 0;
    float mul = 0;

    if (std::abs(value) > 1)
    {
        int mul_shift;
        mul = std::frexp(value, &mul_shift);
        shift = std::min((int32_t)max_shift, bits - mul_shift);
        mul = mul * std::pow(2.f, shift + mul_shift);
    }
    else if (value == 0)
    {
        mul = 0;
        shift = 0;
    }
    else
    {
        int mul_shift;
        mul = std::frexp(value, &mul_shift);
        shift = std::min(max_shift + mul_shift, bits);
        mul = mul * std::pow(2.f, shift);
        shift -= mul_shift;
    }

    assert(std::abs(mul) < std::pow(2, bits));
    assert(shift >= 0 && shift <= max_shift);
    assert(std::abs(value - mul * std::pow(2, -shift)) <= std::numeric_limits<float>::epsilon());
    return { mul, static_cast<int8_t>(shift) };
}

void quantizer::broadcast_output(hlir::graph &graph, const std::unordered_set<node_opcode> &ops)
{
    auto visitor = make_relay_ir_visitor([&](node &node) {
        if (node.inputs().size() == 1)
        {
            auto it = quant_ranges_.find(node.input_at(0).connection());
            if (it != quant_ranges_.end())
                broadcast_output(node, it->second, ops);
        }
    });
    visitor.visit(graph);
}

void quantizer::broadcast_output(hlir::node &node, const value_range<float> &range, const std::unordered_set<node_opcode> &ops)
{
    if (ops.find(node.runtime_opcode()) != ops.end())
    {
        for (auto &out : node.outputs())
        {
            auto it = quant_ranges_.find(&out);
            if (it != quant_ranges_.end())
                it->second = range;

            for (auto &con : out.connections())
                broadcast_output(con->owner(), range, ops);
        }
    }
}

quantizer::histogram::histogram(value_range<float> range, size_t src_bins, size_t dest_bins)
    : range_(range), optimal_range_(range_)
{
    src_bins_.resize(src_bins);
    dest_bins_.resize(dest_bins);

    auto r = range_.max - range_.min;
    src_bin_interval_ = r / src_bins_.size();
}

void quantizer::histogram::record(xtl::span<const float> data)
{
    for (auto value : data)
    {
        auto r_index = (value - range_.min) / src_bin_interval_;
        auto index = (size_t)std::clamp(r_index, 0.f, (float)src_bins_.size() - 1);
        src_bins_[index]++;
    }
}

void quantizer::histogram::finish()
{
    auto zero_threshold = (size_t)std::clamp((0 - range_.min) / src_bin_interval_, 0.f, (float)src_bins_.size() - 1);
    assert(zero_threshold >= 0 && zero_threshold < src_bins_.size());
    auto min_loss = std::numeric_limits<float>::max();
    std::optional<std::pair<size_t, size_t>> threshold;
    const auto dest_bins = dest_bins_.size();
#if 0
    auto total_freq = std::reduce(src_bins_.begin(), src_bins_.end());
    auto zero_threshold = (size_t)std::clamp((0 - range_.min) / src_bin_interval_, 0.f, (float)src_bins_.size() - 1);
    assert(zero_threshold >= 0 && zero_threshold < src_bins_.size());
    auto min_kld = std::numeric_limits<float>::max();
    std::optional<std::pair<size_t, size_t>> threshold;

    for (size_t lower_threshold = 0; lower_threshold <= zero_threshold; lower_threshold++)
    {
        for (size_t upper_threshold = src_bins_.size(); upper_threshold >= lower_threshold + dest_bins && upper_threshold >= zero_threshold; upper_threshold--)
        {
            auto src_range = upper_threshold - lower_threshold;
            auto src_per_bin = (float)src_range / dest_bins;

            std::vector<float> range_dist(src_bins_.begin() + lower_threshold, src_bins_.begin() + upper_threshold);
#if KLD_METHOD == 1
            range_dist.front() += std::reduce(src_bins_.begin(), src_bins_.begin() + lower_threshold);
            range_dist.back() += std::reduce(src_bins_.begin() + upper_threshold, src_bins_.end());
#endif

            // ref dist
            std::vector<float> ref_dist(range_dist);
            ref_dist.front() += std::reduce(src_bins_.begin(), src_bins_.begin() + lower_threshold);
            ref_dist.back() += std::reduce(src_bins_.begin() + upper_threshold, src_bins_.end());

            // quant dist
            std::vector<float> q_dist(dest_bins);
            for (size_t i = 0; i < dest_bins; i++)
            {
                auto start = i * src_per_bin;
                auto end = start + src_per_bin;
                auto value = 0.f;

                auto left_upper = (size_t)std::ceil(start);
                auto right_lower = (size_t)std::floor(end);
                if (left_upper > start)
                    value += (left_upper - start) * range_dist[left_upper - 1];
                if (right_lower < end)
                    value += (end - right_lower) * range_dist[right_lower];
                value += std::reduce(range_dist.begin() + left_upper, range_dist.begin() + right_lower);
                q_dist[i] = value;
            }

            // upsample quant dist
            std::vector<float> ups_q_dist(src_range);
            for (size_t i = 0; i < dest_bins; i++)
            {
                auto start = i * src_per_bin;
                auto end = start + src_per_bin;
                auto count = 0.f;

                auto left_upper = (size_t)std::ceil(start);
                auto right_lower = (size_t)std::floor(end);
                if (left_upper > start)
                {
                    if (range_dist[left_upper - 1])
                        count += (left_upper - start);
                }
                if (right_lower < end)
                {
                    if (range_dist[right_lower])
                        count += (end - right_lower);
                }

                count += std::count_if(range_dist.begin() + left_upper, range_dist.begin() + right_lower, [](float v) { return v; });
                if (!count)
                    continue;
                auto upsample_value = q_dist[i] / count;
                if (left_upper > start)
                {
                    if (range_dist[left_upper - 1])
                        ups_q_dist[left_upper - 1] += (left_upper - start) * upsample_value;
                }
                if (right_lower < end)
                {
                    if (range_dist[right_lower])
                        ups_q_dist[right_lower] += (end - right_lower) * upsample_value;
                }

                for (size_t j = left_upper; j < right_lower; j++)
                {
                    if (range_dist[j])
                        ups_q_dist[j] += upsample_value;
                }
            }

#if KLD_METHOD == 1
            std::vector<float> ups2_q_dist(src_bins_.size());
            std::copy(ups_q_dist.begin(), ups_q_dist.end(), ups2_q_dist.begin() + lower_threshold);
            auto kld = compute_kld(src_bins_, ups2_q_dist);
#else
            auto kld = compute_kld(ref_dist, ups_q_dist);
#endif
            if (kld < min_kld)
            {
                min_kld = kld;
                threshold = { lower_threshold, upper_threshold };
            }
        }
    }
#else
    for (size_t lower_threshold = 0; lower_threshold <= zero_threshold; lower_threshold++)
    {
        for (size_t upper_threshold = src_bins_.size(); upper_threshold >= lower_threshold + dest_bins && upper_threshold >= zero_threshold; upper_threshold--)
        {
            auto dest_min = lower_threshold * src_bin_interval_ + range_.min;
            auto dest_max = upper_threshold * src_bin_interval_ + range_.min;

            auto loss = compute_l2(src_bins_, range_, { dest_min, dest_max }, dest_bins);
            if (loss < min_loss)
            {
                min_loss = loss;
                threshold = { lower_threshold, upper_threshold };
            }
        }
    }
#endif

    assert(threshold);
    if (threshold)
    {
        auto opt_min = threshold->first * src_bin_interval_ + range_.min;
        auto opt_max = threshold->second * src_bin_interval_ + range_.min;
        optimal_range_ = { opt_min, opt_max };
    }

    std::cout << "{" << range_.min << ", " << range_.max << "} -> {" << optimal_range_.min << ", " << optimal_range_.max << "}" << std::endl;
}
