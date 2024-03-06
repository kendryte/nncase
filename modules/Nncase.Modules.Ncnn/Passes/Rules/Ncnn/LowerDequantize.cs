// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using NetFabric.Hyperlinq;
using Nncase.IR;
using Nncase.IR.Math;
using Nncase.IR.Ncnn;
using Nncase.PatternMatch;

using static Nncase.IR.F.Ncnn;
using static Nncase.IR.F.Tensors;
using static Nncase.IR.TypePatternUtility;
using static Nncase.PatternMatch.F.Math;
using static Nncase.PatternMatch.Utility;

namespace Nncase.Passes.Rules.Ncnn;

[RuleGenerator]
public partial class LowerDequantize : RewriteRule<Pattern>
{
    /// <inheritdoc/>
    public override Pattern Pattern { get; } = IsDequantize(
        "deq",
        "output",
        _ => true,
        IsWildcard("input"),
        IsTensorConst("param"));

    private Expr? GetReplace(Expr input, Dequantize deq, QuantParam[] param)
    {
        if (input.CheckedShape.Count > 4 || input.CheckedShape[0].FixedValue != 1 || deq.TargetType != DataTypes.Float32)
        {
            Console.WriteLine("ncnn not support more than 4D or batchSize > 1");
            return null;
        }

        var inRes = Squeeze(input, new[] { 0 });
        var inResO = new Var(inRes.CheckedType);

        float[] scale = new float[param.Length];
        float[] bias = new float[param.Length];
        for (int i = 0; i < param.Length; i++)
        {
            scale[i] = param[i].Scale;
            bias[i] = param[i].ZeroPoint;
        }

        var dequantize = new Call(new Fusion("ncnn", NcnnDequantize(inResO, scale, bias), new[] { inResO }), inRes);

        return Unsqueeze(dequantize, new[] { 0 });
    }
}
