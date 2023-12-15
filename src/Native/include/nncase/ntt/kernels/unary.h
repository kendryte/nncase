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
#include <cmath>

namespace nncase::ntt {
// math ops
namespace mathops {
struct abs {
    float operator()(float v) const noexcept { return fabs(v); }
};

struct acos {
    float operator()(float v) const noexcept { return acosf(v); }
};

struct acosh {
    float operator()(float v) const noexcept { return acoshf(v); }
};

struct asin {
    float operator()(float v) const noexcept { return asinf(v); }
};

struct asinh {
    float operator()(float v) const noexcept { return asinhf(v); }
};

struct ceil {
    float operator()(float v) const noexcept { return ceilf(v); }
};

struct cos {
    float operator()(float v) const noexcept { return cosf(v); }
};

struct cosh {
    float operator()(float v) const noexcept { return coshf(v); }
};

struct exp {
    float operator()(float v) const noexcept { return expf(v); }
};

struct floor {
    float operator()(float v) const noexcept { return floorf(v); }
};

struct log {
    float operator()(float v) const noexcept { return logf(v); }
};

struct neg {
    float operator()(float v) const noexcept { return -v; }
};

struct round {
    float operator()(float v) const noexcept { return nearbyintf(v); }
};

struct sign {
    float operator()(float v) const noexcept { return copysignf(1.f, v); }
};

struct sin {
    float operator()(float v) const noexcept { return sinf(v); }
};

struct sinh {
    float operator()(float v) const noexcept { return sinhf(v); }
};

struct sqrt {
    float operator()(float v) const noexcept { return sqrtf(v); }
};

struct square {
    float operator()(float v) const noexcept { return v * v; }
};

struct tanh {
    float operator()(float v) const noexcept { return tanhf(v); }
};
} // namespace mathops

template <class Op, class TVA, class TVB> void unary(TVA input, TVB output) {
    Op op;
    for (size_t i = 0; i < input.buffer().size(); i++) {
        output.buffer()[i] = op(input.buffer()[i]);
    }
}
} // namespace nncase::ntt
