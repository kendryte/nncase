// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using DryIoc.ImTools;
using Nncase.ArgsStruct;
using Nncase.IR;
using Nncase.IR.Math;
using Nncase.IR.Ncnn;
using Nncase.IR.Tensors;
using Nncase.PatternMatch;

using static Nncase.IR.F.Ncnn;
using static Nncase.IR.F.Tensors;
using static Nncase.IR.TypePatternUtility;
using static Nncase.PatternMatch.F.NN;
using static Nncase.PatternMatch.F.Tensors;
using static Nncase.PatternMatch.Utility;

namespace Nncase.Passes.Rules.Ncnn;

[RuleGenerator]
public partial class LowerCast : RewriteRule<Pattern>
{
    /// <inheritdoc/>
    public override Pattern Pattern { get; } = IsCast(
        "cast",
        _ => true,
        IsWildcard("input") with { TypePattern = HasFixedShape() });

    private int GetNcnnType(DataType inType)
    {
        if (inType == DataTypes.Float32)
        {
            return 1;
        }
        else if (inType == DataTypes.Float16)
        {
            return 2;
        }
        else if (inType == DataTypes.BFloat16)
        {
            return 4;
        }
        else
        {
            return -1;
        }
    }

    private Expr? GetReplace(Expr input, Cast cast)
    {
        if (input.CheckedShape.Count > 4 || input.CheckedShape[0].FixedValue != 1)
        {
            Console.WriteLine("ncnn not support more than 4D or batchSize > 1");
            return null;
        }

        // ncnn only support f32->f16, f32->bf16, f16->f32, bf16->f32
        int fromType = GetNcnnType(input.CheckedDataType);
        int toType = GetNcnnType(cast.NewType);

        if (fromType == -1 || toType == -1)
        {
            return null;
        }

        var inRes = Squeeze(input, new[] { 0 });
        var inResO = new Var(inRes.CheckedType);
        var c = new Call(new Fusion("ncnn", NcnnCast(inResO, fromType, toType), new[] { inResO }), inRes);
        return Unsqueeze(c, new[] { 0 });
    }
}
