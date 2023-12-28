/* Copyright 2019-2021 Canaan Inc.
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
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <nncase/ir/ops/constant.h>
#include <nncase/ir/quantizer.h>
#include <nncase/ir/visitor.h>

using namespace nncase;
using namespace nncase::ir;

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

static std::vector<float> smooth_distribution(const std::vector<float> p, const float eps = 0.0001)
{
    std::vector<size_t> is_zeros(p.size());
    std::vector<size_t> is_nonzeros(p.size());
    {
        auto it = p.begin();
        std::generate(is_zeros.begin(), is_zeros.end(),
            [&it]() { return static_cast<size_t>(*(it++) == 0.f); });
    }
    {
        auto it = p.begin();
        std::generate(is_nonzeros.begin(), is_nonzeros.end(),
            [&it]() { return static_cast<size_t>(*(it++) != 0.f); });
    }
    size_t n_zeros = std::accumulate(is_zeros.begin(), is_zeros.end(), 0);
    size_t n_nonzeros = p.size() - n_zeros;
    if (!n_nonzeros)
    {
        // The discrete probability distribution is malformed. All entries are 0.
        return std::vector<float>();
    }
    float eps1 = eps * static_cast<float>(n_zeros) / static_cast<float>(n_nonzeros);
    if (eps1 >= 1.0)
        return std::vector<float>();
    auto ret = p;
    for (size_t i = 0; i < p.size(); i++)
    {
        ret[i] += eps * is_zeros[i] - eps1 * is_nonzeros[i];
    }
    return ret;
}

static std::vector<float> smooth(const std::vector<float> &p, const size_t box_pts = 512)
{
    std::vector<float> ret(p.size());

    std::vector<float> p_expand(box_pts - 1, 0);
    p_expand.insert(p_expand.end(), p.begin(), p.end());
    p_expand.insert(p_expand.end(), box_pts - 1, 0);

    for (size_t i = box_pts / 2; i < ret.size() + box_pts / 2; i++)
        ret[i - box_pts / 2] = std::reduce(p_expand.begin() + i, p_expand.begin() + i + box_pts) / box_pts;

    return ret;
}

static std::vector<float> calc_cdf(std::vector<float> &p_)
{
    auto p = smooth_distribution(p_);
    auto p_sum = std::reduce(p.begin(), p.end());
    for (auto &value : p)
        value = value / p_sum;

    std::vector<float> cdf(p.size(), 0.f);
    cdf[0] = p[0];
    for (size_t i = 1; i < p.size(); i++)
        cdf[i] = cdf[i - 1] + p[i];

    return cdf;
}

float compute_kld(xtl::span<float> p, xtl::span<float> q)
{
    if (!(p.size() && q.size()) || p.size() != q.size())
        return std::numeric_limits<float>::max();

    auto p_sum = std::reduce(p.begin(), p.end());
    auto q_sum = std::reduce(q.begin(), q.end());
    for (auto &value : p)
        value = value / p_sum;
    for (auto &value : q)
        value = value / q_sum;

    float d = 0.f;
    for (size_t i = 0; i < p.size(); i++)
    {
        if (p[i])
            d += q[i] ? p[i] * std::log(p[i] / q[i]) : 1.f;
    }

    return d;
}

float compute_l2(std::vector<float> &p, value_range<float> p_range, value_range<float> q_range, size_t q_bins)
{
    auto p_interval = (p_range.max - p_range.min) / p.size();
    auto q_interval = (q_range.max - q_range.min) / (q_bins - 1);
    float d = 0.f;

    for (size_t i = 0; i < p.size(); i++)
    {
        auto p_val = p_range.min + p_interval * (i + 0.0f);
        auto q_idx = std::clamp((int32_t)std::round((p_val - q_range.min) / q_interval), 0, (int32_t)q_bins - 1);
        auto q_val = q_range.min + q_interval * (q_idx + 0.0f);
        d += std::pow(p_val - q_val, 2.f) * p[i];
    }

    return d;
}

void run_kld_m2(std::vector<float> &src_bins_, std::optional<std::pair<size_t, size_t>> &threshold, size_t &zero_threshold, const unsigned long &dest_bins)
{
    auto src_bins = smooth(src_bins_);
    auto min_kld = std::numeric_limits<float>::max();

    auto kld = [&](size_t lower_threshold, size_t upper_threshold) {
        auto src_range = upper_threshold - lower_threshold;
        auto src_per_bin = src_range / dest_bins;

        std::vector<float> range_dist(src_bins.begin() + lower_threshold, src_bins.begin() + upper_threshold);

        // ref dist
        std::vector<float> ref_dist(range_dist);
        ref_dist.front() += std::reduce(src_bins.begin(), src_bins.begin() + lower_threshold);
        ref_dist.back() += std::reduce(src_bins.begin() + upper_threshold, src_bins.end());

        // quant dist
        std::vector<float> q_dist(dest_bins);
        for (size_t i = 0; i < dest_bins; i++)
        {
            auto start = i * src_per_bin;
            auto end = start + src_per_bin;
            auto value = 0.f;

            value += std::reduce(ref_dist.begin() + start, ref_dist.begin() + end);
            q_dist[i] = value;
        }

        // upsample quant dist
        std::vector<float> ups_q_dist(src_range);
        for (size_t i = 0; i < dest_bins; i++)
        {
            auto start = i * src_per_bin;
            auto end = start + src_per_bin;
            auto count = 0.f;

            count += std::count_if(ref_dist.begin() + start, ref_dist.begin() + end, [](float v) { return v; });
            if (!count)
                continue;
            auto upsample_value = q_dist[i] / count;

            for (size_t j = start; j < end; j++)
            {
                if (ref_dist[j])
                    ups_q_dist[j] += upsample_value;
            }
        }

        float kld = 0.f;
        std::vector<float> ups2_q_dist(src_bins.size());
        // left outliers
        auto count = 0.f;
        count += std::count_if(src_bins.begin(), src_bins.begin() + lower_threshold + src_per_bin, [](float v) { return v; });
        auto value = std::reduce(src_bins.begin(), src_bins.begin() + lower_threshold + src_per_bin) / count;
        for (size_t i = 0; i < lower_threshold + src_per_bin; i++)
        {
            if (src_bins[i])
                ups2_q_dist[i] += value;
        }
        // median
        std::copy(ups_q_dist.begin() + src_per_bin, ups_q_dist.end() - src_per_bin, ups2_q_dist.begin() + lower_threshold + src_per_bin);
        // right outliers
        count = 0.f;
        count += std::count_if(src_bins.begin() + upper_threshold - src_per_bin, src_bins.end(), [](float v) { return v; });
        value = std::reduce(src_bins.begin() + upper_threshold - src_per_bin, src_bins.end()) / count;
        for (size_t i = upper_threshold - src_per_bin; i < src_bins.size(); i++)
        {
            if (src_bins[i])
                ups2_q_dist[i] += value;
        }

        src_bins = smooth_distribution(src_bins);
        ups2_q_dist = smooth_distribution(ups2_q_dist);
        kld = compute_kld(src_bins_, ups2_q_dist);

        if (kld < min_kld)
        {
            min_kld = kld;
            threshold = { lower_threshold, upper_threshold };
        }
    };

    // range max fisrt
    {
        min_kld = std::numeric_limits<float>::max();
        size_t lower_threshold = 0;
        for (size_t upper_threshold = src_bins.size(); upper_threshold >= dest_bins && upper_threshold >= zero_threshold; upper_threshold -= dest_bins)
        {
            kld(lower_threshold, upper_threshold);
        }
    }
    // range min
    {
        min_kld = std::numeric_limits<float>::max();
        size_t upper_threshold = threshold->second;
        for (size_t lower_threshold = 0; lower_threshold <= zero_threshold && lower_threshold <= upper_threshold - dest_bins; lower_threshold += dest_bins)
        {
            kld(lower_threshold, upper_threshold);
        }
    }
}

void run_l2(std::vector<float> &src_bins_, std::optional<std::pair<size_t, size_t>> &threshold, size_t zero_threshold, float src_bin_interval_, value_range<float> range_, const unsigned long dest_bins)
{
    auto min_loss = std::numeric_limits<float>::max();

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
}

void run_cdf(std::vector<float> &src_bins_, std::optional<std::pair<size_t, size_t>> &threshold, size_t &zero_threshold, const unsigned long &dest_bins)
{
    auto slope_threshold = 0.001f;
    auto cdf = calc_cdf(src_bins_);

    size_t lower_threshold = 0;
    for (; lower_threshold <= zero_threshold; lower_threshold++)
    {
        if (cdf[lower_threshold] / (lower_threshold + 1) * src_bins_.size() > slope_threshold)
            break;
    }
    size_t upper_threshold = src_bins_.size() - 1;
    for (; upper_threshold >= lower_threshold + dest_bins && upper_threshold >= zero_threshold; upper_threshold--)
    {
        if ((1 - cdf[upper_threshold]) / (src_bins_.size() - upper_threshold) * src_bins_.size() > slope_threshold)
            break;
    }

    threshold = { lower_threshold, upper_threshold };
}
}

quantizer::quantizer([[maybe_unused]] calibrate_method cali_method, size_t bins)
    : cali_method_(cali_method), bins_(bins)
{
}

void quantizer::record(ir::output_connector &connector, value_range<float> range)
{
    auto it = quant_ranges_.find(&connector);
    if (it == quant_ranges_.end())
        quant_ranges_.emplace(&connector, range);
    else
        it->second = union_range(it->second, range);
}

void quantizer::set(ir::output_connector &connector, value_range<float> range)
{
    quant_ranges_[&connector] = range;
}

bool quantizer::has_record(ir::output_connector &connector) const
{
    return has_record_.contains(&connector) && has_record_.at(&connector);
}

void quantizer::record(output_connector &connector, std::span<const float> data)
{
    switch (stage_)
    {
    case quantize_stage::collect_range:
        record(connector, get_range(data.begin(), data.end()));
        has_record_.emplace(&connector, true);
        if (std::find(ranges_insert_order_.begin(), ranges_insert_order_.end(), &connector) == ranges_insert_order_.end())
            ranges_insert_order_.push_back(&connector);
        break;
    case quantize_stage::collect_distribution:
        if (connector.owner().runtime_opcode() != op_constant)
            histograms_.at(&connector).record(data);
        has_record_.emplace(&connector, true);
        break;
    default:
        throw std::runtime_error("Invalid operation in current quantization stage");
    }
}

void quantizer::record(output_connector &connector, std::span<const bfloat16> data)
{
    switch (stage_)
    {
    case quantize_stage::collect_range:
        record(connector, get_range(data.begin(), data.end()));
        has_record_.emplace(&connector, true);
        if (std::find(ranges_insert_order_.begin(), ranges_insert_order_.end(), &connector) == ranges_insert_order_.end())
            ranges_insert_order_.push_back(&connector);
        break;
    case quantize_stage::collect_distribution:
        if (connector.owner().runtime_opcode() != op_constant)
            histograms_.at(&connector).record(data);
        has_record_.emplace(&connector, true);
        break;
    default:
        throw std::runtime_error("Invalid operation in current quantization stage");
    }
}

void quantizer::record(output_connector &connector, std::span<const half> data)
{
    switch (stage_)
    {
    case quantize_stage::collect_range:
        record(connector, get_range(data.begin(), data.end()));
        has_record_.emplace(&connector, true);
        if (std::find(ranges_insert_order_.begin(), ranges_insert_order_.end(), &connector) == ranges_insert_order_.end())
            ranges_insert_order_.push_back(&connector);
        break;
    case quantize_stage::collect_distribution:
        if (connector.owner().runtime_opcode() != op_constant)
            histograms_.at(&connector).record(data);
        has_record_.emplace(&connector, true);
        break;
    default:
        throw std::runtime_error("Invalid operation in current quantization stage");
    }
}

void quantizer::record_buffers(output_connector &connector, std::span<const float> data)
{
    std::vector<float> data_vec;
    data_vec.assign(data.begin(), data.end());
    output_buffers_.emplace(&connector, data_vec);
}

void quantizer::record_buffers(output_connector &connector, std::span<const bfloat16> data)
{
    std::vector<float> data_vec;
    for (int i = 0; i < data.size(); i++)
        data_vec.push_back((float)(data.data()[i]));
    output_buffers_.emplace(&connector, data_vec);
}

void quantizer::record_buffers(output_connector &connector, std::span<const half> data)
{
    std::vector<float> data_vec;
    for (int i = 0; i < data.size(); i++)
        data_vec.push_back((float)(data.data()[i]));
    output_buffers_.emplace(&connector, data_vec);
}

void quantizer::record_quant_buffers(output_connector &connector, std::span<const float> data)
{
    std::vector<float> data_vec;
    data_vec.assign(data.begin(), data.end());
    output_buffers_.emplace(&connector, data_vec);
    if (std::find(quant_buffers_insert_order_.begin(), quant_buffers_insert_order_.end(), &connector) == quant_buffers_insert_order_.end())
        quant_buffers_insert_order_.push_back(&connector);
}

void quantizer::record_quant_buffers(output_connector &connector, std::span<const bfloat16> data)
{
    std::vector<float> data_vec;
    for (int i = 0; i < data.size(); i++)
        data_vec.push_back(static_cast<float>(data.data()[i]));
    output_buffers_.emplace(&connector, data_vec);
    if (std::find(quant_buffers_insert_order_.begin(), quant_buffers_insert_order_.end(), &connector) == quant_buffers_insert_order_.end())
        quant_buffers_insert_order_.push_back(&connector);
}

void quantizer::record_quant_buffers(output_connector &connector, std::span<const half> data)
{
    std::vector<float> data_vec;
    for (int i = 0; i < data.size(); i++)
        data_vec.push_back(static_cast<float>(data.data()[i]));
    output_buffers_.emplace(&connector, data_vec);
    if (std::find(quant_buffers_insert_order_.begin(), quant_buffers_insert_order_.end(), &connector) == quant_buffers_insert_order_.end())
        quant_buffers_insert_order_.push_back(&connector);
}

void quantizer::begin_collect_distribution()
{
    for (auto &&p : quant_ranges_)
    {
        if (p.first->owner().runtime_opcode() != op_constant)
            histograms_.emplace(p.first, histogram(fixup_range(p.second), bins_, 256, cali_method_));
    }

    stage_ = quantize_stage::collect_distribution;
}

void quantizer::end_collect_distribution(std::function<void(size_t cnt, size_t total)> progress)
{
    size_t i = 0;
    for (auto &&h : histograms_)
    {
        std::cout << h.first->owner().name() << std::endl;
        h.second.finish();
        quant_ranges_.at(h.first) = h.second.optimal_range();
        if (progress)
            progress(i++, histograms_.size());
    }
}

quant_param_t quantizer::get_quant_param(value_range<float> range, int32_t bits, quant_mode qm)
{
    if (qm == quant_mode::signed_symmetric_mode)
        range = fixup_range(range, true);
    else
        range = fixup_range(range);
    double Q_max = 255;
    double Q_min = 0;
    switch (qm)
    {
    case quant_mode::unsigned_mode:
        Q_min = 0;
        Q_max = (1 << bits) - 1;
        break;
    case quant_mode::signed_symmetric_mode:
        Q_min = -(1 << (bits - 1)) + 1;
        Q_max = (1 << (bits - 1)) - 1;
        break;
    case quant_mode::signed_asymmetric_mode:
        Q_min = -(1 << (bits - 1));
        Q_max = (1 << (bits - 1)) - 1;
        break;
    default:
        throw std::runtime_error("Invalid quant mode");
    }
    auto scale = (range.max - range.min) / (Q_max - Q_min);
    auto bias = std::round((range.min * (Q_min - Q_max)) / (range.max - range.min)) + Q_min;
    return { static_cast<int32_t>(bias), (float)scale };
}

value_range<float> quantizer::get(ir::output_connector &connector) const
{
    return quant_ranges_.at(&connector);
}

fixed_mul quantizer::get_fixed_mul(float value, int32_t max_bits, uint8_t max_shift, bool is_signed)
{
    assert(is_signed || value >= 0);

    auto bits = is_signed ? max_bits - 1 : max_bits;
    int32_t shift = 0;
    float mul = 0;

    if (value == 0)
    {
        mul = 0;
        shift = 0;
    }
    else if (std::abs(value) > 1)
    {
        int mul_shift;
        mul = std::frexp(value, &mul_shift);
        shift = std::min((int32_t)max_shift, bits - mul_shift);
        mul = mul * std::pow(2.f, (float)(shift + mul_shift));
    }
    else
    {
        int mul_shift;
        mul = std::frexp(value, &mul_shift);
        shift = std::min(max_shift + mul_shift, bits);
        mul = mul * std::pow(2.f, (float)shift);
        shift -= mul_shift;
    }

    assert(std::abs(mul) < std::pow(2.f, (float)bits));
    assert(shift >= 0 && shift <= max_shift);
    assert(std::abs(value - mul * std::pow(2.f, (float)-shift)) <= std::numeric_limits<float>::epsilon());
    return { mul, static_cast<int8_t>(shift) };
}

void quantizer::set_model_output_range(ir::graph &graph)
{
    auto visitor = make_relay_ir_visitor([&](node &node) {
        if (node.runtime_opcode() == op_output_node)
        {
            auto it = quant_ranges_.find(node.input_at(0).connection());
            if (it != quant_ranges_.end())
                model_output_range_ = it->second;
            else
                throw std::runtime_error("Can't get model output range!");
        } });
    visitor.visit(graph);
}

void quantizer::broadcast_output(ir::graph &graph, const std::unordered_set<node_opcode> &ops)
{
    auto visitor = make_relay_ir_visitor([&](node &node) {
        if (node.inputs().size() == 1)
        {
            auto it = quant_ranges_.find(node.input_at(0).connection());
            if (it != quant_ranges_.end())
                broadcast_output(node, it->second, ops);
        } });
    visitor.visit(graph);
}

void quantizer::broadcast_output(ir::node &node, const value_range<float> &range, const std::unordered_set<node_opcode> &ops)
{
    if (ops.find(node.runtime_opcode()) != ops.end())
    {
        for (auto out : node.outputs())
        {
            auto it = quant_ranges_.find(out);
            if (it != quant_ranges_.end())
                it->second = range;

            for (auto con : out->connections())
                broadcast_output(con->owner(), range, ops);
        }
    }
}

quantizer::histogram::histogram(value_range<float> range, size_t src_bins, size_t dest_bins, calibrate_method cali_method)
    : range_(range), optimal_range_(range_), cali_method_(cali_method)
{
    src_bins_.resize(src_bins);
    dest_bins_.resize(dest_bins);

    auto r = range_.max - range_.min;
    src_bin_interval_ = r / src_bins_.size();
}
void quantizer::histogram::record(std::span<const bfloat16> data)
{
    for (auto value : data)
    {
        auto r_index = (value - range_.min) / src_bin_interval_;
        auto index = (size_t)std::clamp(r_index, 0.f, (float)src_bins_.size() - 1);
        src_bins_[index]++;
    }
}
void quantizer::histogram::record(std::span<const float> data)
{
    for (auto value : data)
    {
        auto r_index = (value - range_.min) / src_bin_interval_;
        auto index = (size_t)std::clamp(r_index, 0.f, (float)src_bins_.size() - 1);
        src_bins_[index]++;
    }
}
void quantizer::histogram::record(std::span<const half> data)
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
    std::string quant_method_name = "";
    auto zero_threshold = (size_t)std::clamp((0 - range_.min) / src_bin_interval_, 0.f, (float)src_bins_.size() - 1);
    assert(zero_threshold < src_bins_.size());
    std::optional<std::pair<size_t, size_t>> threshold;
    const auto dest_bins = dest_bins_.size();

    if (cali_method_ == calibrate_method::kld_m0 || cali_method_ == calibrate_method::kld_m1)
    {
        auto min_kld = std::numeric_limits<float>::max();

        for (size_t lower_threshold = 0; lower_threshold <= zero_threshold; lower_threshold++)
        {
            for (size_t upper_threshold = src_bins_.size(); upper_threshold >= lower_threshold + dest_bins && upper_threshold >= zero_threshold; upper_threshold--)
            {
                auto src_range = upper_threshold - lower_threshold;
                auto src_per_bin = (float)src_range / dest_bins;

                std::vector<float> range_dist(src_bins_.begin() + lower_threshold, src_bins_.begin() + upper_threshold);

                // ref dist
                std::vector<float> ref_dist(range_dist);
                ref_dist.front() += std::reduce(src_bins_.begin(), src_bins_.begin() + lower_threshold);
                ref_dist.back() += std::reduce(src_bins_.begin() + upper_threshold, src_bins_.end());

                if (cali_method_ == calibrate_method::kld_m1)
                {
                    range_dist.front() += std::reduce(src_bins_.begin(), src_bins_.begin() + lower_threshold);
                    range_dist.back() += std::reduce(src_bins_.begin() + upper_threshold, src_bins_.end());
                }

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
                        if (ref_dist[left_upper - 1])
                            ups_q_dist[left_upper - 1] += (left_upper - start) * upsample_value;
                    }
                    if (right_lower < end)
                    {
                        if (ref_dist[right_lower])
                            ups_q_dist[right_lower] += (end - right_lower) * upsample_value;
                    }

                    for (size_t j = left_upper; j < right_lower; j++)
                    {
                        if (ref_dist[j])
                            ups_q_dist[j] += upsample_value;
                    }
                }

                float kld = 0.f;
                if (cali_method_ == calibrate_method::kld_m1)
                {
                    std::vector<float> ups2_q_dist(src_bins_.size());
                    std::copy(ups_q_dist.begin(), ups_q_dist.end(), ups2_q_dist.begin() + lower_threshold);
                    src_bins_ = smooth_distribution(src_bins_);
                    ups2_q_dist = smooth_distribution(ups2_q_dist);
                    kld = compute_kld(src_bins_, ups2_q_dist);
                }
                else
                {
                    ref_dist = smooth_distribution(ref_dist);
                    ups_q_dist = smooth_distribution(ups_q_dist);
                    kld = compute_kld(ref_dist, ups_q_dist);
                }
                if (kld < min_kld)
                {
                    min_kld = kld;
                    threshold = { lower_threshold, upper_threshold };
                }
            }
        }
    }
    else if (cali_method_ == calibrate_method::kld_m2)
    {
        run_kld_m2(src_bins_, threshold, zero_threshold, dest_bins);
    }
    else if (cali_method_ == calibrate_method::l2)
    {
        run_l2(src_bins_, threshold, zero_threshold, src_bin_interval_, range_, dest_bins);
    }
    else if (cali_method_ == calibrate_method::cdf)
    {
        run_cdf(src_bins_, threshold, zero_threshold, dest_bins);
    }
    else if (cali_method_ == calibrate_method::auto_select)
    {

        std::vector<calibrate_method> method_list { calibrate_method::kld_m2, calibrate_method::l2, calibrate_method::no_clip };
        auto min_loss = std::numeric_limits<float>::max();
        auto new_threshold = threshold;

        for (auto i : method_list)
        {
            std::cout << "range_: " << range_.min << ", " << range_.max << std::endl;
            std::string tmp_name = "";
            if (i == calibrate_method::kld_m2)
            {
                run_kld_m2(src_bins_, new_threshold, zero_threshold, dest_bins);
                tmp_name = "kld_m2";
            }
            else if (i == calibrate_method::l2)
            {
                run_l2(src_bins_, new_threshold, zero_threshold, src_bin_interval_, range_, dest_bins);
                tmp_name = "l2";
            }
            else if (i == calibrate_method::cdf)
            {
                run_cdf(src_bins_, new_threshold, zero_threshold, dest_bins);
                tmp_name = "cdf";
            }
            else
            {
                new_threshold = { size_t(0), size_t(src_bins_.size() - 1) };
                tmp_name = "no_clip";
            }
            auto opt_min = (new_threshold->first - 0.5f) * src_bin_interval_ + range_.min;
            auto opt_max = (new_threshold->second + 0.5f) * src_bin_interval_ + range_.min;

            value_range<float> tmp_range = { opt_min, opt_max };
            if (i == calibrate_method::no_clip)
                tmp_range = range_;
            auto new_loss = compute_l2(src_bins_, range_, tmp_range, dest_bins);
            if (new_loss < min_loss)
            {
                min_loss = new_loss;
                threshold = new_threshold;
                quant_method_name = tmp_name;
            }
        }
    }

    assert(threshold);
    if (threshold)
    {
        auto opt_min = (threshold->first - 0.5f) * src_bin_interval_ + range_.min;
        auto opt_max = (threshold->second + 0.5f) * src_bin_interval_ + range_.min;
        if (quant_method_name == "no_clip")
            optimal_range_ = range_;
        else
            optimal_range_ = { opt_min, opt_max };
    }
    if (quant_method_name != "")
        std::cout << "[ " << quant_method_name << " ]" << std::endl;
    std::cout << "{" << range_.min << ", " << range_.max << "} -> {" << optimal_range_.min << ", " << optimal_range_.max << "}" << std::endl;
}
