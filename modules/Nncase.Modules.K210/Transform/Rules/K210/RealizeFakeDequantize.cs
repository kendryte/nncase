// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Nncase.IR;
using Nncase.IR.F;
using Nncase.IR.K210;
using Nncase.IR.Math;
using Nncase.PatternMatch;
using Nncase.Utilities;
using Tensorflow.Keras;
using static Nncase.IR.TypePatternUtility;
using static Nncase.PatternMatch.F.K210;
using static Nncase.PatternMatch.F.Math;
using static Nncase.PatternMatch.F.NN;
using static Nncase.PatternMatch.Utility;
using static Nncase.Quantization.Utility;
using Math = Nncase.IR.F.Math;

namespace Nncase.Passes.Rules.K210;

/// <summary>
/// Lower <see cref="FakeDequantize"/> to <see cref="Dequantize"/>.
/// </summary>
[RuleGenerator]
public sealed partial class RealizeFakeDequantize : IRewriteRule
{
    /// <inheritdoc/>
    public IPattern Pattern { get; } = IsFakeDequantize(
            null,
            "dequant_call",
            op => true,
            IsWildcard("input"),
            IsQuantParamOf("quantparam", op => true, IsConst("range"), IsConst("bits")) with { TypePattern = HasFixedShape() });

    private Expr? GetReplace(Call dequant_call, Expr input, Expr quantparam, Tensor<float> range, int bits)
    {
        // For k210, mode is Unsigned, bits is 8.
        var qm = QuantMode.UnsignedMode;
        var qp = QuantUtility.GetQuantParam((range[0], range[1]), bits, qm);
        qp.Scale = 1f / qp.Scale;
        return IR.F.Math.Dequantize(input, qp, DataTypes.Float32);
    }
}
