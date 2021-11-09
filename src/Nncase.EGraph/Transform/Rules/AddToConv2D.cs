using System;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.Linq;
using Nncase.Transform.Pattern.NN;
using Nncase.Transform.Pattern.Tensors;
using static Nncase.Transform.Pattern.Utility;
using static Nncase.IR.F.NN;
using static Nncase.IR.F.Tensors;
using static Nncase.Transform.Pattern.F.Math;
using static Nncase.Transform.Pattern.F.Tensors;
using System.Numerics.Tensors;
using Nncase.IR;
using Nncase.Transform.Pattern.Math;

namespace Nncase.Transform.Rule;


public sealed class AddToConv2D : EGraphRule
{
    private BinaryWrapper ad;

    public AddToConv2D()
    {
        Pattern = ad = Add(IsWildCard(), IsWildCard());
    }

    public override Expr? GetRePlace(EMatchResult result)
    {
        ad.Bind(result);
        var a_sp = ad.Lhs().CheckedShape;
        var b_sp = ad.Rhs().CheckedShape;
        if (a_sp.Rank == 4 && a_sp == b_sp)
        {
            var channels = a_sp[1].FixedValue;
            var weights = new DenseTensor<float>(channels * 2 * channels);
            for (int i = 0; i < channels; i++)
            {
                weights[2 * channels * i + i] = 1.0f;
                weights[2 * channels * i + i + channels] = 1.0f;
            }
            var c = Concat(new IR.Tuple(ad.Lhs(), ad.Rhs()), 1);
            var con_weights = Const.FromTensor<float>(weights.Reshape(new[] { channels, 2 * channels, 1, 1 }));

            return Conv2D(c, con_weights,
              Enumerable.Repeat(0.0f, channels).ToArray(),
              new DenseTensor<int>(new[] { 0, 0, 0, 0 }, new[] { 2, 2 }),
              new[] { 1, 1 },
              new[] { 1, 1 },
              PadMode.Constant,
              1);
        }
        return null;
    }
}
