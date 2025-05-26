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
#pragma once
#include "../apply.h"
#include "nncase/ntt/primitive_ops.h"
#include "nncase/ntt/shape.h"
#include "u_mul_add.h"

namespace nncase::ntt {
namespace ukernels {
namespace detail {
template <bool TransposedB, size_t N0Tile> struct b_stride_getter;

template <size_t N0Tile> struct b_stride_getter<true, N0Tile> {
    using Stride = fixed_shape<N0Tile, 1>;
};

template <size_t N0Tile> struct b_stride_getter<false, N0Tile> {
    using Stride = fixed_shape<1, N0Tile>;
};
} // namespace detail

template <mamtul_pack_kind PackKind, class TLhsElem, class TRhsElem,
          class TOutElem, bool Arch>
struct u_matmul_policy {
    static constexpr size_t m0_tile = 1;
    static constexpr size_t n0_tile = 1;
    static constexpr size_t m0_subtile = 0;
};

template <mamtul_pack_kind PackKind, class TA, class TB, class TC>
struct u_type_scale {
    using TLhsElem = std::decay_t<typename TA::element_type>;
    using TRhsElem = std::decay_t<typename TB::element_type>;
    using TOutElem = std::decay_t<typename TC::element_type>;
    static constexpr size_t m0_scale = 1;
    static constexpr size_t n0_scale = 1;
    static constexpr size_t k0_scale = 1;
    static constexpr bool same_type = false;
};

template <class TA, class TB, class TC>
struct u_type_scale<ukernels::mamtul_pack_kind::no_pack, TA, TB, TC> {
    using TLhsElem = std::decay_t<typename TA::element_type>;
    using TRhsElem = std::decay_t<typename TB::element_type>;
    using TOutElem = std::decay_t<typename TC::element_type>;
    static constexpr size_t m0_scale = 1;
    static constexpr size_t n0_scale = 1;
    static constexpr size_t k0_scale = 1;
    static constexpr bool same_type = std::is_same_v<TLhsElem, TOutElem>;
};

template <class TA, class TB, class TC>
struct u_type_scale<ukernels::mamtul_pack_kind::pack_k, TA, TB, TC> {
    using TLhsElem = std::decay_t<typename TA::element_type>::element_type;
    using TRhsElem = std::decay_t<typename TB::element_type>;
    using TOutElem = std::decay_t<typename TC::element_type>;
    static constexpr size_t m0_scale = 1;
    static constexpr size_t n0_scale = 1;
    static constexpr size_t k0_scale = sizeof(TOutElem) / sizeof(TLhsElem);
    static constexpr bool same_type = std::is_same_v<TLhsElem, TOutElem>;
};

template <class TA, class TB, class TC>
struct u_type_scale<ukernels::mamtul_pack_kind::pack_m, TA, TB, TC> {
    using TLhsElem = std::decay_t<typename TA::element_type>;
    using TRhsElem = std::decay_t<typename TB::element_type>;
    using TOutElem = std::decay_t<typename TC::element_type>;
    static constexpr size_t m0_scale = (TLhsElem::size()) / (TOutElem::size());
    static constexpr size_t n0_scale = 1;
    static constexpr size_t k0_scale = 1;
    static constexpr bool same_type =
        std::is_same_v<typename TLhsElem::element_type,
                       typename TOutElem::element_type>;
};

template <class TA, class TB, class TC>
struct u_type_scale<ukernels::mamtul_pack_kind::pack_n, TA, TB, TC> {
    using TLhsElem = std::decay_t<typename TA::element_type>;
    using TRhsElem = std::decay_t<typename TB::element_type>;
    using TOutElem = std::decay_t<typename TC::element_type>;
    static constexpr size_t m0_scale = 1;
    static constexpr size_t n0_scale = (TRhsElem::size()) / (TOutElem::size());
    static constexpr size_t k0_scale = 1;
    static constexpr bool same_type =
        std::is_same_v<typename TRhsElem::element_type,
                       typename TOutElem::element_type>;
};

template <class TA, class TB, class TC>
struct u_type_scale<ukernels::mamtul_pack_kind::pack_mk, TA, TB, TC> {
    using TLhsElem = std::decay_t<typename TA::element_type>;
    using TRhsElem = std::decay_t<typename TB::element_type>;
    using TOutElem = std::decay_t<typename TC::element_type>;
    static constexpr size_t m0_scale =
        (TLhsElem::size()) / (TLhsElem::shape()[1] * (TOutElem::size()));
    static constexpr size_t n0_scale = 1;
    static constexpr size_t k0_scale = m0_scale;
    static constexpr bool same_type =
        std::is_same_v<typename TRhsElem::element_type,
                       typename TOutElem::element_type>;
};

template <class TA, class TB, class TC>
struct u_type_scale<ukernels::mamtul_pack_kind::pack_kn, TA, TB, TC> {
    using TLhsElem = std::decay_t<typename TA::element_type>;
    using TRhsElem = std::decay_t<typename TB::element_type>;
    using TOutElem = std::decay_t<typename TC::element_type>;
    static constexpr size_t m0_scale = 1;
    static constexpr size_t n0_scale =
        (TRhsElem::size()) / (TRhsElem::shape()[0] * (TOutElem::size()));
    static constexpr size_t k0_scale = n0_scale;
    static constexpr bool same_type =
        std::is_same_v<typename TRhsElem::element_type,
                       typename TOutElem::element_type>;
};

template <class TA, class TB, class TC>
struct u_type_scale<ukernels::mamtul_pack_kind::pack_mn, TA, TB, TC> {
    using TLhsElem = std::decay_t<typename TA::element_type>;
    using TRhsElem = std::decay_t<typename TB::element_type>;
    using TOutElem = std::decay_t<typename TC::element_type>;
    static constexpr size_t m0_scale =
        TLhsElem::shape()[0] / TOutElem::shape()[0];
    static constexpr size_t n0_scale =
        TLhsElem::shape()[0] / TOutElem::shape()[1];
    static constexpr size_t k0_scale = 1;
    static constexpr bool same_type =
        std::is_same_v<typename TRhsElem::element_type,
                       typename TOutElem::element_type>;
};

template <class TA, class TB, class TC>
struct u_type_scale<ukernels::mamtul_pack_kind::pack_mkn, TA, TB, TC> {
    using TLhsElem = std::decay_t<typename TA::element_type>;
    using TRhsElem = std::decay_t<typename TB::element_type>;
    using TOutElem = std::decay_t<typename TC::element_type>;
    static constexpr size_t m0_scale =
        TLhsElem::shape()[0] / TOutElem::shape()[0];
    static constexpr size_t n0_scale =
        TLhsElem::shape()[0] / TOutElem::shape()[1];
    static constexpr size_t k0_scale = n0_scale;
    static constexpr bool same_type =
        std::is_same_v<typename TRhsElem::element_type,
                       typename TOutElem::element_type>;
};

template <ukernels::mamtul_pack_kind PackKind, bool AccumulateC,
          bool TransposedA, bool TransposedB, size_t M0Tile, size_t N0Tile,
          class TLhsElem, class TRhsElem, class TOutElem, bool Arch>
struct u_matmul_generic {
    using BStride = detail::b_stride_getter<TransposedB, N0Tile>::Stride;

    template <class TA, class TB, class TC>
    constexpr void operator()(const TA &a, const TB &b, TC &c0,
                              size_t K) noexcept {

        constexpr auto m0_scale =
            ukernels::u_type_scale<PackKind, TA, TB, TC>::m0_scale;
        constexpr auto n0_scale =
            ukernels::u_type_scale<PackKind, TA, TB, TC>::n0_scale;
        constexpr auto same_type =
            ukernels::u_type_scale<PackKind, TA, TB, TC>::same_type;
        constexpr auto m0_tile_scaled = m0_scale * M0Tile;
        constexpr auto n0_tile_scaled = n0_scale * N0Tile;

        TOutElem c0_tmp[m0_tile_scaled][n0_tile_scaled];
        ntt::apply(c0.shape(), [&](auto index) {
            c0_tmp[index[0]][index[1]] = AccumulateC ? c0(index) : TOutElem{};
        });

        using TLhsElemExpanded =
            cast_fixed_tensor_element_type<TOutElem, float>::type;
        TLhsElemExpanded c0_grouped[m0_tile_scaled][n0_tile_scaled];
        if constexpr (IsScalar<TLhsElemExpanded>) {
            for (size_t i = 0; i < m0_tile_scaled; i++) {
                for (size_t j = 0; j < n0_tile_scaled; j++) {
                    c0_grouped[i][j] = (float)c0_tmp[i][j];
                }
            }
        } else {
            for (size_t i = 0; i < m0_tile_scaled; i++) {
                for (size_t j = 0; j < n0_tile_scaled; j++) {
                    ntt::apply(c0_grouped[i][j].shape(), [&](auto index) {
                        c0_grouped[i][j](index) = (float)c0_tmp[i][j](index);
                    });
                }
            }
        }

        for (size_t k1 = 0; k1 < K; k1++) {
            auto a0 =
                a.view(make_ranked_shape(0, k1), fixed_shape<M0Tile, 1>{});
            auto b0 = b.view(TransposedB ? make_ranked_shape(0, k1)
                                         : make_ranked_shape(k1, 0),
                             BStride{});
            TLhsElem a0_tmp[M0Tile];
            TRhsElem b0_tmp[N0Tile];

            ntt::apply(fixed_shape<M0Tile>{},
                       [&](auto index) { a0_tmp[index[0]] = a0(index[0], 0); });
            if constexpr (TransposedB) {
                ntt::apply(fixed_shape<N0Tile>{}, [&](auto index) {
                    b0_tmp[index[0]] = b0(index[0], 0);
                });
            } else {
                ntt::apply(fixed_shape<N0Tile>{}, [&](auto index) {
                    b0_tmp[index[0]] = b0(0, index[0]);
                });
            }

            if constexpr ((ukernels::mamtul_pack_kind::pack_k == PackKind) &&
                          (!same_type)) {
                using TLhsElemExpanded =
                    cast_fixed_tensor_element_type<TLhsElem, float>::type;
                using TLhsElemGrouped = TLhsElemExpanded;

                using TRhsElemExpanded =
                    cast_fixed_tensor_element_type<TRhsElem, float>::type;
                using TRhsElemGrouped = TRhsElemExpanded;

                TLhsElemGrouped a0_grouped[M0Tile];
                TRhsElemGrouped b0_grouped[N0Tile];

                loop<M0Tile>([&](auto i) {
                    ntt::apply(a0_grouped[i].shape(), [&](auto index) {
                        a0_grouped[i](index) = (float)a0_tmp[i](index);
                    });
                });

                loop<N0Tile>([&](auto i) {
                    ntt::apply(b0_grouped[i].shape(), [&](auto index) {
                        b0_grouped[i](index) = (float)b0_tmp[i](index);
                    });
                });

                for (size_t n = 0; n < N0Tile; n++) {
                    for (size_t m = 0; m < M0Tile; m++) {
                        u_mul_add<PackKind, true>(a0_grouped[m], b0_grouped[n],
                                                  c0_grouped[m][n]);
                    }
                }

                loop<m0_tile_scaled>([&](auto i) {
                    loop<n0_tile_scaled>([&](auto j) {
                        c0_tmp[i][j] = (TOutElem)c0_grouped[i][j];
                    });
                });

            } else if constexpr ((ukernels::mamtul_pack_kind::pack_m ==
                                  PackKind) &&
                                 (!same_type)) {
                using TLhsElemExpanded =
                    cast_fixed_tensor_element_type<TLhsElem, float>::type;
                using TLhsElemGrouped =
                    ntt::fixed_tensor_alike_t<TLhsElemExpanded, m0_scale,
                                              TLhsElemExpanded::size() /
                                                  m0_scale>;

                using TRhsElemGrouped = float;
                TLhsElemGrouped a0_grouped[M0Tile];
                TRhsElemGrouped b0_grouped[N0Tile];
                loop<M0Tile>([&](auto i) {
                    ntt::apply(a0_grouped[i].shape(), [&](auto index) {
                        a0_grouped[i](index) = (float)a0_tmp[i](
                            index[0] * a0_grouped[i].shape()[1] + index[1]);
                    });
                });
                loop<N0Tile>([&](auto i) { b0_grouped[i] = (float)b0_tmp[i]; });

                for (size_t n = 0; n < N0Tile; n++) {
                    for (size_t m = 0; m < M0Tile; m++) {
                        for (size_t k = 0; k < m0_scale; k++) {
                            u_mul_add<PackKind, true>(a0_grouped[m](k),
                                                      b0_grouped[n],
                                                      c0_grouped[k][n]);
                        }
                    }
                }

                using TElem = TOutElem::element_type;
                loop<m0_tile_scaled>([&](auto i) {
                    loop<n0_tile_scaled>([&](auto j) {
                        ntt::apply(c0_tmp[i][j].shape(), [&](auto index) {
                            c0_tmp[i][j](index) =
                                (TElem)c0_grouped[i][j](index);
                        });
                    });
                });

            } else if constexpr ((ukernels::mamtul_pack_kind::pack_n ==
                                  PackKind) &&
                                 (!same_type)) {
                using TRhsElemExpanded =
                    cast_fixed_tensor_element_type<TRhsElem, float>::type;
                using TRhsElemGrouped =
                    ntt::fixed_tensor_alike_t<TRhsElemExpanded, n0_scale,
                                              TRhsElemExpanded::size() /
                                                  n0_scale>;

                using TLhsElemGrouped = float;
                TLhsElemGrouped a0_grouped[M0Tile];
                TRhsElemGrouped b0_grouped[N0Tile];
                loop<M0Tile>([&](auto i) { a0_grouped[i] = (float)a0_tmp[i]; });
                loop<N0Tile>([&](auto i) {
                    ntt::apply(b0_grouped[i].shape(), [&](auto index) {
                        b0_grouped[i](index) = (float)b0_tmp[i](
                            index[0] * b0_grouped[i].shape()[1] + index[1]);
                    });
                });

                for (size_t n = 0; n < N0Tile; n++) {
                    for (size_t m = 0; m < M0Tile; m++) {
                        for (size_t k = 0; k < n0_scale; k++) {
                            u_mul_add<PackKind, true>(a0_grouped[m],
                                                      b0_grouped[n](k),
                                                      c0_grouped[m][k]);
                        }
                    }
                }

                using TElem = TOutElem::element_type;
                loop<m0_tile_scaled>([&](auto i) {
                    loop<n0_tile_scaled>([&](auto j) {
                        ntt::apply(c0_tmp[i][j].shape(), [&](auto index) {
                            c0_tmp[i][j](index) =
                                (TElem)c0_grouped[i][j](index);
                        });
                    });
                });

            } else if constexpr ((ukernels::mamtul_pack_kind::pack_mk ==
                                  PackKind) &&
                                 (!same_type)) {
                using TLhsElemExpanded =
                    cast_fixed_tensor_element_type<TLhsElem, float>::type;
                using TLhsElemGrouped =
                    ntt::fixed_tensor_alike_t<TLhsElemExpanded, m0_scale,
                                              TLhsElemExpanded::shape()[0] /
                                                  m0_scale,
                                              TLhsElemExpanded::shape()[1]>;

                using TRhsElemExpanded =
                    cast_fixed_tensor_element_type<TRhsElem, float>::type;
                using TRhsElemGrouped = TRhsElemExpanded;

                TLhsElemGrouped a0_grouped[M0Tile];
                TRhsElemGrouped b0_grouped[N0Tile];
                loop<M0Tile>([&](auto i) {
                    ntt::apply(a0_grouped[i].shape(), [&](auto index) {
                        a0_grouped[i](index) = (float)a0_tmp[i](
                            index[0] * a0_grouped[i].shape()[1] + index[1],
                            index[2]);
                    });
                });
                loop<N0Tile>([&](auto i) {
                    ntt::apply(b0_grouped[i].shape(), [&](auto index) {
                        b0_grouped[i](index) = (float)b0_tmp[i](index);
                    });
                });

                for (size_t n = 0; n < N0Tile; n++) {
                    for (size_t m = 0; m < M0Tile; m++) {
                        for (size_t k = 0; k < m0_scale; k++) {
                            u_mul_add<PackKind, true>(a0_grouped[m](k),
                                                      b0_grouped[n],
                                                      c0_grouped[k][n]);
                        }
                    }
                }

                using TElem = TOutElem::element_type;
                loop<m0_tile_scaled>([&](auto i) {
                    loop<n0_tile_scaled>([&](auto j) {
                        ntt::apply(c0_tmp[i][j].shape(), [&](auto index) {
                            c0_tmp[i][j](index) =
                                (TElem)c0_grouped[i][j](index);
                        });
                    });
                });

            } else if constexpr ((ukernels::mamtul_pack_kind::pack_mn ==
                                  PackKind) &&
                                 (!same_type)) {
                using TLhsElemExpanded =
                    cast_fixed_tensor_element_type<TLhsElem, float>::type;
                using TLhsElemGrouped =
                    ntt::fixed_tensor_alike_t<TLhsElemExpanded, m0_scale,
                                              TLhsElemExpanded::size() /
                                                  m0_scale>;

                using TRhsElemExpanded =
                    cast_fixed_tensor_element_type<TRhsElem, float>::type;
                using TRhsElemGrouped =
                    ntt::fixed_tensor_alike_t<TRhsElemExpanded, n0_scale,
                                              TRhsElemExpanded::size() /
                                                  n0_scale>;

                TLhsElemGrouped a0_grouped[M0Tile];
                TRhsElemGrouped b0_grouped[N0Tile];

                loop<M0Tile>([&](auto i) {
                    ntt::apply(a0_grouped[i].shape(), [&](auto index) {
                        a0_grouped[i](index) = (float)a0_tmp[i](
                            index[0] * a0_grouped[i].shape()[1] + index[1]);
                    });
                });
                loop<N0Tile>([&](auto i) {
                    ntt::apply(b0_grouped[i].shape(), [&](auto index) {
                        b0_grouped[i](index) = (float)b0_tmp[i](
                            index[0] * b0_grouped[i].shape()[1] + index[1]);
                    });
                });

                for (size_t n = 0; n < N0Tile; n++) {
                    for (size_t m = 0; m < M0Tile; m++) {
                        for (size_t k = 0; k < n0_scale; k++) {
                            for (size_t l = 0; l < m0_scale; l++) {
                                u_mul_add<PackKind, true>(a0_grouped[m](l),
                                                          b0_grouped[n](k),
                                                          c0_grouped[l][k]);
                            }
                        }
                    }
                }

                using TElem = TOutElem::element_type;
                loop<m0_tile_scaled>([&](auto i) {
                    loop<n0_tile_scaled>([&](auto j) {
                        ntt::apply(c0_tmp[i][j].shape(), [&](auto index) {
                            c0_tmp[i][j](index) =
                                (TElem)c0_grouped[i][j](index);
                        });
                    });
                });

            } else if constexpr ((ukernels::mamtul_pack_kind::pack_mkn ==
                                  PackKind) &&
                                 (!same_type)) {
                using TLhsElemExpanded =
                    cast_fixed_tensor_element_type<TLhsElem, float>::type;
                using TLhsElemGrouped =
                    ntt::fixed_tensor_alike_t<TLhsElemExpanded, m0_scale,
                                              TLhsElemExpanded::shape()[0] /
                                                  m0_scale,
                                              TLhsElemExpanded::shape()[1]>;

                using TRhsElemExpanded =
                    cast_fixed_tensor_element_type<TRhsElem, float>::type;
                using TRhsElemGrouped = ntt::fixed_tensor_alike_t<
                    TRhsElemExpanded, n0_scale, TRhsElemExpanded::shape()[0],
                    TRhsElemExpanded::shape()[1] / n0_scale>;

                TLhsElemGrouped a0_grouped[M0Tile];
                TRhsElemGrouped b0_grouped[N0Tile];
                loop<M0Tile>([&](auto i) {
                    ntt::apply(a0_grouped[i].shape(), [&](auto index) {
                        a0_grouped[i](index) = (float)a0_tmp[i](
                            index[0] * a0_grouped[i].shape()[1] + index[1],
                            index[2]);
                    });
                });
                loop<N0Tile>([&](auto i) {
                    ntt::apply(b0_grouped[i].shape(), [&](auto index) {
                        b0_grouped[i](index) = (float)b0_tmp[i](
                            index[1],
                            index[0] * b0_grouped[i].shape()[2] + index[2]);
                    });
                });

                for (size_t n = 0; n < N0Tile; n++) {
                    for (size_t m = 0; m < M0Tile; m++) {
                        for (size_t k = 0; k < n0_scale; k++) {
                            for (size_t l = 0; l < m0_scale; l++) {
                                u_mul_add<PackKind, true>(a0_grouped[m](l),
                                                          b0_grouped[n](k),
                                                          c0_grouped[l][k]);
                            }
                        }
                    }
                }

                using TElem = TOutElem::element_type;
                loop<m0_tile_scaled>([&](auto i) {
                    loop<n0_tile_scaled>([&](auto j) {
                        ntt::apply(c0_tmp[i][j].shape(), [&](auto index) {
                            c0_tmp[i][j](index) =
                                (TElem)c0_grouped[i][j](index);
                        });
                    });
                });

            } else if constexpr ((ukernels::mamtul_pack_kind::no_pack ==
                                  PackKind) &&
                                 (!same_type)) {
                float a0_grouped[M0Tile];
                float b0_grouped[N0Tile];

                loop<M0Tile>([&](auto i) { a0_grouped[i] = (float)a0_tmp[i]; });
                loop<N0Tile>([&](auto i) { b0_grouped[i] = (float)b0_tmp[i]; });

                for (size_t n = 0; n < N0Tile; n++) {
                    for (size_t m = 0; m < M0Tile; m++) {
                        u_mul_add<PackKind, true>(a0_grouped[m], b0_grouped[n],
                                                  c0_grouped[m][n]);
                    }
                }

                loop<m0_tile_scaled>([&](auto i) {
                    loop<n0_tile_scaled>([&](auto j) {
                        c0_tmp[i][j] = (TOutElem)c0_grouped[i][j];
                    });
                });

            } else {
                for (size_t n = 0; n < N0Tile; n++) {
                    for (size_t m = 0; m < M0Tile; m++) {
                        u_mul_add<PackKind, true>(a0_tmp[m], b0_tmp[n],
                                                  c0_tmp[m][n]);
                    }
                }
            }
        }

        ntt::apply(c0.shape(), [&](auto index) {
            ntt::store(c0(index), c0_tmp[index[0]][index[1]]);
        });
    }
};

template <ukernels::mamtul_pack_kind PackKind, bool AccumulateC,
          bool TransposedA, bool TransposedB, size_t M0Tile, size_t N0Tile,
          class TLhsElem, class TRhsElem, class TOutElem, bool Arch>
struct u_matmul
    : u_matmul_generic<PackKind, AccumulateC, TransposedA, TransposedB, M0Tile,
                       N0Tile, TLhsElem, TRhsElem, TOutElem, Arch> {};

template <bool AccumulateC, bool TransposedA, bool TransposedB, size_t M0Tile,
          size_t N0Tile, class TLhsElem, class TRhsElem, class TOutElem,
          bool Arch>
struct u_matmul<ukernels::mamtul_pack_kind::pack_mn, AccumulateC, TransposedA,
                TransposedB, M0Tile, N0Tile, TLhsElem, TRhsElem, TOutElem,
                Arch> {
    using BStride = detail::b_stride_getter<TransposedB, N0Tile>::Stride;
    template <class TA, class TB, class TC>
    constexpr void operator()(const TA &a, const TB &b, TC &c0,
                              size_t K) noexcept {
        using TSubOutElem = ntt::vector<typename TOutElem::element_type,
                                        TOutElem::shape().last()>;
        using policy_t =
            ntt::ukernels::u_matmul_policy<mamtul_pack_kind::pack_mn, TLhsElem,
                                           TRhsElem, TOutElem, true>;
        constexpr auto m0_subtile = policy_t::m0_subtile;

        if constexpr (m0_subtile) {
            TSubOutElem c0_tmp[m0_subtile][N0Tile];

            for (size_t sm1 = 0; sm1 < TOutElem::shape()[0];
                 sm1 += m0_subtile) {
                ntt::apply(fixed_shape<m0_subtile, N0Tile>{}, [&](auto index) {
                    c0_tmp[index[0]][index[1]] =
                        AccumulateC ? c0(0, index[1])(sm1 + index[0])
                                    : TSubOutElem{};
                });

                for (size_t k1 = 0; k1 < K; k1++) {
                    using TSubLhsElem = typename TLhsElem::element_type;
                    TSubLhsElem a0_tmp[m0_subtile];
                    TRhsElem b0_tmp[N0Tile];

                    auto a0 = a.view(make_ranked_shape(0, k1),
                                     fixed_shape<M0Tile, 1>{});
                    auto b0 = b.view(TransposedB ? make_ranked_shape(0, k1)
                                                 : make_ranked_shape(k1, 0),
                                     BStride{});

                    ntt::apply(fixed_shape<m0_subtile>{}, [&](auto index) {
                        a0_tmp[index[0]] = a0(0, 0)(sm1 + index[0]);
                    });
                    if constexpr (TransposedB) {
                        ntt::apply(fixed_shape<N0Tile>{}, [&](auto index) {
                            b0_tmp[index[0]] = b0(index[0], 0);
                        });
                    } else {
                        ntt::apply(fixed_shape<N0Tile>{}, [&](auto index) {
                            b0_tmp[index[0]] = b0(0, index[0]);
                        });
                    }

                    for (size_t n = 0; n < N0Tile; n++) {
                        for (size_t m = 0; m < m0_subtile; m++) {
                            c0_tmp[m][n] = ntt::mul_add(a0_tmp[m], b0_tmp[n],
                                                        c0_tmp[m][n]);
                        }
                    }
                }

                ntt::apply(fixed_shape<m0_subtile, N0Tile>{}, [&](auto index) {
                    ntt::store(c0(0, index[1])(sm1 + index[0]),
                               c0_tmp[index[0]][index[1]]);
                });
            }
        } else {
            u_matmul_generic<mamtul_pack_kind::pack_mn, AccumulateC,
                             TransposedA, TransposedB, M0Tile, N0Tile, TLhsElem,
                             TRhsElem, TOutElem, Arch>
                impl;
            impl(a, b, c0, K);
        }
    }
};

template <bool AccumulateC, bool TransposedA, bool TransposedB, size_t M0Tile,
          size_t N0Tile, class TLhsElem, class TRhsElem, class TOutElem,
          bool Arch>
struct u_matmul<ukernels::mamtul_pack_kind::pack_kn, AccumulateC, TransposedA,
                TransposedB, M0Tile, N0Tile, TLhsElem, TRhsElem, TOutElem,
                Arch> {
    using BStride = detail::b_stride_getter<TransposedB, N0Tile>::Stride;
    template <class TA, class TB, class TC>
    constexpr void operator()(const TA &a, const TB &b, TC &c0,
                              size_t K) noexcept {
        constexpr auto m0_scale =
            ukernels::u_type_scale<ukernels::mamtul_pack_kind::pack_kn, TA, TB,
                                   TC>::m0_scale;
        constexpr auto n0_scale =
            ukernels::u_type_scale<ukernels::mamtul_pack_kind::pack_kn, TA, TB,
                                   TC>::n0_scale;
        constexpr auto same_type =
            ukernels::u_type_scale<ukernels::mamtul_pack_kind::pack_kn, TA, TB,
                                   TC>::same_type;
        constexpr auto m0_tile_scaled = m0_scale * M0Tile;
        constexpr auto n0_tile_scaled = n0_scale * N0Tile;

        TOutElem c0_tmp[m0_tile_scaled][n0_tile_scaled];
        ntt::apply(c0.shape(), [&](auto index) {
            c0_tmp[index[0]][index[1]] = AccumulateC ? c0(index) : TOutElem{};
        });

        using TLhsElemExpanded =
            cast_fixed_tensor_element_type<TOutElem, float>::type;
        TLhsElemExpanded c0_grouped[m0_tile_scaled][n0_tile_scaled];
        if constexpr (IsScalar<TLhsElemExpanded>) {
            for (size_t i = 0; i < m0_tile_scaled; i++) {
                for (size_t j = 0; j < n0_tile_scaled; j++) {
                    c0_grouped[i][j] = (float)c0_tmp[i][j];
                }
            }
        } else {
            for (size_t i = 0; i < m0_tile_scaled; i++) {
                for (size_t j = 0; j < n0_tile_scaled; j++) {
                    ntt::apply(c0_grouped[i][j].shape(), [&](auto index) {
                        c0_grouped[i][j](index) = (float)c0_tmp[i][j](index);
                    });
                }
            }
        }

        for (size_t k1 = 0; k1 < K; k1++) {
            auto a0 =
                a.view(make_ranked_shape(0, k1), fixed_shape<M0Tile, 1>{});
            auto b0 = b.view(TransposedB ? make_ranked_shape(0, k1)
                                         : make_ranked_shape(k1, 0),
                             BStride{});
            for (size_t sk1 = 0; sk1 < TLhsElem::shape()[0]; sk1++) {
                using TSubLhsElem = typename TLhsElem::element_type;
                using TSubRhsElem = ntt::vector<typename TRhsElem::element_type,
                                                TRhsElem::shape().last()>;

                TSubLhsElem a0_tmp[M0Tile];
                TSubRhsElem b0_tmp[N0Tile];
                ntt::apply(fixed_shape<M0Tile>{}, [&](auto index) {
                    a0_tmp[index[0]] = a0(index[0], 0)(sk1);
                });
                if constexpr (TransposedB) {
                    ntt::apply(fixed_shape<N0Tile>{}, [&](auto index) {
                        b0_tmp[index[0]] = b0(index[0], 0)(sk1);
                    });
                } else {
                    ntt::apply(fixed_shape<N0Tile>{}, [&](auto index) {
                        b0_tmp[index[0]] = b0(0, index[0])(sk1);
                    });
                }

                if constexpr (!same_type) {

                    using TLhsElemExpanded = float;
                    using TLhsElemGrouped = TLhsElemExpanded;

                    using TRhsElemExpanded =
                        cast_fixed_tensor_element_type<TSubRhsElem,
                                                       float>::type;
                    using TRhsElemGrouped = ntt::fixed_tensor_alike_t<
                        TRhsElemExpanded, n0_scale,
                        (TRhsElemExpanded::shape()[0] / n0_scale)>;

                    TLhsElemGrouped a0_grouped[M0Tile];
                    TRhsElemGrouped b0_grouped[N0Tile];

                    loop<M0Tile>(
                        [&](auto i) { a0_grouped[i] = (float)a0_tmp[i]; });

                    loop<N0Tile>([&](auto i) {
                        ntt::apply(b0_grouped[i].shape(), [&](auto index) {
                            b0_grouped[i](index) = (float)b0_tmp[i](
                                index[0] * b0_grouped[i].shape()[1] + index[1]);
                        });
                    });

                    for (size_t n = 0; n < N0Tile; n++) {
                        for (size_t m = 0; m < M0Tile; m++) {
                            for (size_t k = 0; k < n0_scale; k++) {
                                c0_grouped[m][k] = ntt::mul_add(
                                    a0_grouped[m], b0_grouped[n](k),
                                    c0_grouped[m][k]);
                            }
                        }
                    }

                    using TElem = TOutElem::element_type;
                    loop<m0_tile_scaled>([&](auto i) {
                        loop<n0_tile_scaled>([&](auto j) {
                            ntt::apply(c0_tmp[i][j].shape(), [&](auto index) {
                                c0_tmp[i][j](index) =
                                    (TElem)c0_grouped[i][j](index);
                            });
                        });
                    });

                } else {
                    for (size_t n = 0; n < N0Tile; n++) {
                        for (size_t m = 0; m < M0Tile; m++) {
                            c0_tmp[m][n] = ntt::mul_add(a0_tmp[m], b0_tmp[n],
                                                        c0_tmp[m][n]);
                        }
                    }
                }
            }
        }

        ntt::apply(c0.shape(), [&](auto index) {
            ntt::store(c0(index), c0_tmp[index[0]][index[1]]);
        });
    }
};

template <bool AccumulateC, bool TransposedA, bool TransposedB, size_t M0Tile,
          size_t N0Tile, class TLhsElem, class TRhsElem, class TOutElem,
          bool Arch>
struct u_matmul<ukernels::mamtul_pack_kind::pack_mkn, AccumulateC, TransposedA,
                TransposedB, M0Tile, N0Tile, TLhsElem, TRhsElem, TOutElem,
                Arch> {
    using BStride = detail::b_stride_getter<TransposedB, N0Tile>::Stride;
    template <class TA, class TB, class TC>
    constexpr void operator()(const TA &a, const TB &b, TC &c0,
                              size_t K) noexcept {
        using TSubOutElem = ntt::vector<typename TOutElem::element_type,
                                        TOutElem::shape().last()>;
        using policy_t =
            ntt::ukernels::u_matmul_policy<mamtul_pack_kind::pack_mkn, TLhsElem,
                                           TRhsElem, TOutElem, true>;
        constexpr auto m0_subtile = policy_t::m0_subtile;

        if constexpr (m0_subtile) {
            TSubOutElem c0_tmp[m0_subtile][N0Tile];

            for (size_t sm1 = 0; sm1 < TOutElem::shape()[0];
                 sm1 += m0_subtile) {
                ntt::apply(fixed_shape<m0_subtile, N0Tile>{}, [&](auto index) {
                    c0_tmp[index[0]][index[1]] =
                        AccumulateC ? c0(0, index[1])(sm1 + index[0])
                                    : TSubOutElem{};
                });

                for (size_t k1 = 0; k1 < K; k1++) {
                    // Force compiler do not unroll the loop
                    size_t sk1_max = TLhsElem::shape()[1];
#pragma GCC unroll 1
                    for (size_t sk1 = 0; sk1 < sk1_max; sk1++) {
                        using TSubLhsElem = typename TLhsElem::element_type;
                        using TSubRhsElem =
                            ntt::vector<typename TRhsElem::element_type,
                                        TRhsElem::shape().last()>;

                        auto a0 = a.view(make_ranked_shape(0, k1),
                                         fixed_shape<M0Tile, 1>{});
                        auto b0 = b.view(TransposedB ? make_ranked_shape(0, k1)
                                                     : make_ranked_shape(k1, 0),
                                         BStride{});

                        TSubLhsElem a0_tmp[m0_subtile];
                        TSubRhsElem b0_tmp[N0Tile];

                        ntt::apply(fixed_shape<m0_subtile>{}, [&](auto index) {
                            a0_tmp[index[0]] = a0(0, 0)(sm1 + index[0], sk1);
                        });
                        if constexpr (TransposedB) {
                            ntt::apply(fixed_shape<N0Tile>{}, [&](auto index) {
                                b0_tmp[index[0]] = b0(index[0], 0)(sk1);
                            });
                        } else {

                            ntt::apply(fixed_shape<N0Tile>{}, [&](auto index) {
                                b0_tmp[index[0]] = b0(0, index[0])(sk1);
                            });
                        }

                        for (size_t n = 0; n < N0Tile; n++) {
                            for (size_t m = 0; m < m0_subtile; m++) {
                                auto &output = c0_tmp[m][n];
                                output =
                                    ntt::mul_add(a0_tmp[m], b0_tmp[n], output);
                            }
                        }
                    }
                }

                ntt::apply(fixed_shape<m0_subtile, N0Tile>{}, [&](auto index) {
                    ntt::store(c0(0, index[1])(sm1 + index[0]),
                               c0_tmp[index[0]][index[1]]);
                });
            }
        } else {
            u_matmul_generic<mamtul_pack_kind::pack_mkn, AccumulateC,
                             TransposedA, TransposedB, M0Tile, N0Tile, TLhsElem,
                             TRhsElem, TOutElem, Arch>
                impl;
            impl(a, b, c0, K);
        }
    }
};
} // namespace ukernels

template <ukernels::mamtul_pack_kind PackKind, bool AccumulateC,
          bool TransposedA, bool TransposedB, size_t M0Tile, size_t N0Tile,
          class TA, class TB, class TC>
constexpr void u_matmul(const TA &a, const TB &b, TC &c, size_t K) noexcept {
    using TLhsElem = std::decay_t<typename TA::element_type>;
    using TRhsElem = std::decay_t<typename TB::element_type>;
    using TOutElem = std::decay_t<typename TC::element_type>;
    ukernels::u_matmul<PackKind, AccumulateC, TransposedA, TransposedB, M0Tile,
                       N0Tile, TLhsElem, TRhsElem, TOutElem, true>
        impl;
    impl(a, b, c, K);
}
} // namespace nncase::ntt
