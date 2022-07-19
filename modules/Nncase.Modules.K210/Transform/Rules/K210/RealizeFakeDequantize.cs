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
using static  Nncase.PatternMatch.F.K210;
using Nncase.PatternMatch;
using Nncase.Utilities;
using Tensorflow.Keras;
using static Nncase.IR.TypePatternUtility;
using static Nncase.PatternMatch.F.Math;
using static Nncase.PatternMatch.F.NN;
using static Nncase.PatternMatch.Utility;
using Math = Nncase.IR.F.Math;
namespace Nncase.Transform.Rules.K210;

/// <summary>
/// Lower <see cref="IR.K210.FakeDeQuantize"/> to <see cref="IR.K210.DeQuantize"/>.
/// </summary>
[RuleGenerator]
public sealed partial class RealizeFakeDequantize : IRewriteRule
{
    /// <inheritdoc/>
    public IPattern Pattern { get; } = IsFakeDequantize(
            "dequant",
            "dequant_call",
            op => true,
            IsQuantParamOf(op => true, IsConst("range"), IsConst("bits")) with { TypePattern = HasFixedShape() },
            IsTensorConst("quantParam"));

    private Expr? GetReplace(FakeQuantize dequant, Call dequant_call, Expr range, Expr bits, Expr quantParam)
    {
        // For k210, mode is Unsigned, bits is 8.
        return IR.F.Math.Dequantize(range, quantParam, DataTypes.UInt8);
    }
}
