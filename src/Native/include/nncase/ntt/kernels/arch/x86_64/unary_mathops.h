

namespace nncase::ntt::mathops {
template <> struct swish<ntt::vector<float, 8>> {
    ntt::vector<float, 8> operator()(ntt::vector<float, 8> v) const noexcept {
        return impl(v);
    }

    __m256 impl(__m256 v) const noexcept {
        auto zero = _mm256_set1_ps(0);
        auto one = _mm256_set1_ps(1);
        return v / exp256_ps(zero - v) + one;
    }
};

template <> struct neg<ntt::vector<float, 8>> {
    ntt::vector<float, 8> operator()(ntt::vector<float, 8> v) const noexcept {
        return _mm256_set1_ps(0) - (__m256)v;
    }
};
} // namespace nncase::ntt::mathops