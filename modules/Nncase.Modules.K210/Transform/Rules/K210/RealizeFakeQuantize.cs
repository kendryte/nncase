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
using static Nncase.PatternMatch.F.NN;
using static Nncase.PatternMatch.Utility;
using Math = Nncase.IR.F.Math;
using static Nncase.PatternMatch.F.Math;
namespace Nncase.Transform.Rules.K210;





/// <summary>
/// Lower <see cref="IR.K210.FakeQuantize"/> to <see cref="IR.K210.Quantize"/>.
/// </summary>
[RuleGenerator]
public sealed partial class RealizeFakeQuantize : IRewriteRule
{
    /// <inheritdoc/>
    public IPattern Pattern { get; } = IsFakeQuantize(
           "quant",
           "quant_call",
           op => true,
           IsWildcard("input") with { TypePattern = HasFixedShape() },
           IsTensorConst("quantParam")) with
           {
               TypePattern = HasFixedShape(),
           };

    private Expr? GetReplace(FakeQuantize quant, Call quant_call, Expr input, Expr quantParam)
    {
        return null;
    }
}