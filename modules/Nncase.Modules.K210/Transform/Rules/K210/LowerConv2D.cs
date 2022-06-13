// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Nncase.IR;
using Nncase.PatternMatch;
using static Nncase.IR.TypePatternUtility;
using static Nncase.PatternMatch.F.NN;
using static Nncase.PatternMatch.Utility;

namespace Nncase.Transform.Rules.K210;

/// <summary>
/// Lower <see cref="IR.NN.Conv2D"/> to <see cref="IR.K210.FakeKPUConv2D"/>.
/// </summary>
[RuleGenerator]
public sealed class LowerConv2D : IRewriteRule
{
    /// <inheritdoc/>
    public IPattern Pattern { get; } = IsConv2D(
        PadMode.Constant,
        IsWildcard("input") with { TypePattern = HasFixedShape() },
        IsTensorConst("weights"),
        IsTensorConst("bias"),
        IsTensorConst("strides"),
        IsTensorConst("paddings"),
        new[] { 1, 1 },
        IsTensorConst("groups"),
        IsTensorConst("fusedClamp"));

    private Expr? GetReplace(Expr input, TensorConst weights, Tensor<float> bias, int[] strides, int[] paddings, int groups, float[] fusedClamp)
    {
        if (input.CheckedShape.Equals(newShape.Value.ToArray<int>()))
        {
            return input;
        }

        return null;
    }
}
