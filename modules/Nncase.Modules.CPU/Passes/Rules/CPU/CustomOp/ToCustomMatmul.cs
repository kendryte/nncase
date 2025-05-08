// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using DryIoc.ImTools;
using NetFabric.Hyperlinq;
using Nncase.CostModel;
using Nncase.IR;
using Nncase.IR.Math;
using Nncase.Passes.Distributed;
using Nncase.PatternMatch;
using Nncase.PatternMatch.F;
using static Nncase.IR.F.Math;
using static Nncase.IR.TypePatternUtility;
using static Nncase.PatternMatch.Utility;

namespace Nncase.Passes.Rules.CPU.CustomOp;

[RuleGenerator]
public partial class ToCustomMatmul : RewriteRule<Pattern>
{
    public ToCustomMatmul(CustomOpScheme scheme)
    {
        Scheme = scheme;
    }

    public ToCustomMatmul()
    {
        Scheme = null!;
    }

    public CustomOpScheme Scheme { get; }

    /// <inheritdoc/>
    public override Pattern Pattern { get; } = Nncase.PatternMatch.F.Math.IsMatMul(
        "mm",
        "call",
        _ => true,
        IsWildcard("lhs"),
        IsWildcard("rhs"));

    private Expr? GetReplace(Call call, MatMul mm, Expr lhs, Expr rhs)
    {
        if (Scheme is null)
        {
            return null;
        }

        // Name pattern
        var node = Scheme.Outputs.FirstOrDefault(op => call.Metadata.OutputNames?[0] is string outputName && outputName == op.Name);

        if (node is null)
        {
            node = Scheme.Outputs.FirstOrDefault(op =>
                op.Op.ToLower(CultureInfo.CurrentCulture) == "matmul" &&
                op.Shape[0].SequenceEqual(lhs.CheckedShape.ToValueArray()) &&
                op.Shape[1].SequenceEqual(rhs.CheckedShape.ToValueArray()));
        }

        if (node is not null)
        {
            return call.With(
                target: new IR.CustomCPU.MatMul(null!, null!, null!, null!, false, false, node!.SBP[0], node!.SBP[1], node!.SBP[2], new() { [CostFactorNames.CPUCycles] = node.Cost }, node.CSourcePath),
                arguments: new[] { lhs, rhs },
                metadata: call.Metadata);
        }

        return null;
    }
}
