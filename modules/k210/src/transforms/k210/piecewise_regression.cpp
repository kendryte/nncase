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
#include <nncase/transforms/k210/piecewise_regression.h>
#include <xtensor/xarray.hpp>
#include <xtensor/xsort.hpp>
#include <xtensor/xtensor.hpp>
//#include <xtensor/

using namespace nncase::ir::transforms::k210;

class dc_regression
{
public:
    dc_regression()
        : y_hat_(0), z_(0), a_(0), b_(0), lanbda_(1.f)
    {
    }

private:
    void auto_tune(xt::xtensor<float, 2> X, float y, uint32_t max_hyper_iter = 10)
    {
        uint32_t n_folds = 5;
        xt::xtensor<float, 1> lanbdas { 1e-3f, 1e-2f, 1e-1f, 1.f, 1e1f, 1e2f, 1e3f };

        for (size_t iter = 0; iter < max_hyper_iter; iter++)
        {
            uint32_t i = 0;
            xt::xtensor<float, 1> loss = xt::zeros<float>(lanbdas.shape());
            for (auto lanbda : lanbdas)
            {
                lanbda_ = lanbda;
                loss[i] = cross_validate(X, y, n_folds);
                i++;
            }

            auto arg_min = xt::argmin(loss)[0];
            auto lanbda = lanbdas[arg_min];
            if (arg_min == 0)
                lanbdas = lanbda * xt::xtensor<float, 1> { 1e-5f, 1e-4f, 1e-3f, 1e-2f, 1e-1f, 1.f, 1e1f };
            else if (arg_min == lanbdas.size() - 1)
                lanbdas = lanbda * xt::xtensor<float, 1> { 1e-1f, 1.f, 1e1f, 1e2f, 1e3f, 1e4f, 1e5f };
            else
            {
                if (lanbdas.size() == 7)
                    lanbdas = lanbda * xt::xtensor<float, 1> { 0.0625f, 0.125f, 0.25f, 0.5f, 1.f, 2.f, 4.f, 8.f, 16.f };
                else
                {
                    lanbda_ = lanbda;
                    fit(X, y, lanbda);
                    break;
                }
            }
        }
    }

    void fit(xt::xtensor<float, 2> X, float y, float lanbda = 0.f, float T = 0.f)
    {
        if (lanbda == 0.f)
        {
            auto_tune(X, y);
            return;
        }
        else
        {
            lanbda_=lanbda;
        }

        auto[n, dim]=X.shape();
        float rho = 0.01f;
        if (T == 0.f)
            T = 2.f * n;
        
        // initial values
        // primal
        xt::xtensor<float, 1> y_hat = xt::zeros<float>(n);
        xt::xtensor<float, 1> z = xt::zeros<float>(n);
        xt::xtensor<float, 2> a = xt::zeros<float>({n,dim});
        xt::xtensor<float, 2> b = xt::zeros<float>({n,dim});
        xt::xtensor<float, 2> p = xt::zeros<float>({n,dim});
        xt::xtensor<float, 2> q = xt::zeros<float>({n,dim});

        xt::xtensor<float, 2> L = xt::zeros<float>({size_t(1),dim});

        // slack
        xt::xtensor<float, 2> s = xt::zeros<float>({n,n});
        xt::xtensor<float, 2> t = xt::zeros<float>({n,n});
        xt::xtensor<float, 2> u = xt::zeros<float>({n,dim});

        // dual
        xt::xtensor<float, 2> alpha = xt::zeros<float>({n,n});
        xt::xtensor<float, 2> beta = xt::zeros<float>({n,n});
        xt::xtensor<float, 2> gamma = xt::zeros<float>({n,dim});
        xt::xtensor<float, 2> eta = xt::zeros<float>({n,dim});
        xt::xtensor<float, 2> zeta = xt::zeros<float>({n,dim});
        
        // preprocess1
        //auto XjXj = xt::dot
    }

private:
    float y_hat_;
    float z_;
    float a_;
    float b_;
    float lanbda_;
};

piecewise_regression::piecewise_regression(size_t segments_count)
    : desired_segments_count_(segments_count)
{
}

std::vector<segment> piecewise_regression::fit(std::vector<point> &points) const
{
    if (points.size() <= desired_segments_count_)
        throw std::invalid_argument("Insufficient points");

    std::sort(points.begin(), points.end(), [](const point &a, const point &b)
        { return a.x < b.x; });

    // 1. initialize segments
    std::vector<segment> segments(points.size() - 1);
    for (size_t i = 0; i < points.size() - 1; i++)
    {
        const auto &p0 = points[i];
        const auto &p1 = points[i + 1];
        segments[i] = { p0.x, p1.x, (p1.y - p0.y) / (p1.x - p0.x), p0.y };
    }

    // 2. combine sibling segments
    while (segments.size() != desired_segments_count_)
    {
        // 2.1 find min slope difference
        float min_diff = std::numeric_limits<float>::max();
        size_t min_idx = -1;
        for (size_t i = 0; i < segments.size() - 1; i++)
        {
            const auto &s0 = segments[i];
            const auto &s1 = segments[i + 1];
            auto diff = std::abs(s0.slop - s1.slop);
            if (diff < min_diff)
            {
                min_diff = diff;
                min_idx = i;
            }
        }

        // 2.2 combine
        auto &s0 = segments[min_idx];
        auto &s1 = segments[min_idx + 1];
        auto y0 = s0.y(s0.start);
        auto y1 = s1.y(s1.stop);
        auto slope = (y1 - y0) / (s1.stop - s0.start);
        s0.slop = slope;
        s0.stop = s1.stop;
        segments.erase(segments.begin() + min_idx + 1);
    }

    return segments;
}
