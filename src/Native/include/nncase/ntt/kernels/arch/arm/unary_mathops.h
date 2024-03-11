

namespace nncase::ntt::mathops {
template <> struct sqrt<ntt::vector<float, 8>> {
    ntt::vector<float, 8> operator()(ntt::vector<float, 8> v) const noexcept {
        float32x4x2_t vv = v;
        return float32x4x2_t { vsqrtq_f32(vv.val[0]), vsqrtq_f32(vv.val[1])};
    }
};
} // namespace nncase::ntt::mathops