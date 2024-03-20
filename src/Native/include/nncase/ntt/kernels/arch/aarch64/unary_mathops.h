

namespace nncase::ntt::mathops {
template <> struct sqrt<ntt::vector<float, 8>> {
    ntt::vector<float, 8> operator()(ntt::vector<float, 8> v) const noexcept {
        float32x4x2_t vv = v;
        return float32x4x2_t{vsqrtq_f32(vv.val[0]), vsqrtq_f32(vv.val[1])};
    }
};

template <> struct swish<ntt::vector<float, 8>> {
    ntt::vector<float, 8> operator()(ntt::vector<float, 8> v) const noexcept {
        float32x4x2_t vv = v;
        return float32x4x2_t{impl(vv.val[0]), impl(vv.val[1])};
    }

    float32x4_t impl(float32x4_t v) const noexcept {
        auto zero = vdupq_n_f32(0);
        auto one = vdupq_n_f32(1);
        return v / exp_ps(zero - v) + one;
    }
};
} // namespace nncase::ntt::mathops