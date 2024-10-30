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
using Nncase.IR.CPU;
using Nncase.IR.Math;
using Nncase.Passes.Distributed;
using Nncase.PatternMatch;
using Nncase.PatternMatch.F;
using static Nncase.IR.F.Math;
using static Nncase.IR.TypePatternUtility;
using static Nncase.PatternMatch.Utility;

namespace Nncase.Passes.Rules.CPU.CustomOp;

[RuleGenerator]
public partial class ToCustomUnary : RewriteRule<Pattern>
{
    public ToCustomUnary(CustomOpScheme scheme)
    {
        Scheme = scheme;
    }

    public ToCustomUnary()
    {
        Scheme = null!;
    }

    public CustomOpScheme Scheme { get; }

    /// <inheritdoc/>
    public override Pattern Pattern { get; } = Nncase.PatternMatch.F.Math.IsUnary(
        "unary",
        "call",
        _ => true,
        IsWildcard("input"));

    private Expr? GetReplace(Call call, Unary unary, Expr input)
    {
        if (Scheme is null)
        {
            return null;
        }

        // Name pattern
        var node = Scheme.Outputs.FirstOrDefault(op => call.Metadata.OutputNames?[0] is string outputName && outputName == op.Name);

        if (node is null)
        {
            node = Scheme.Outputs.FirstOrDefault(op => op.Op.ToLower(CultureInfo.CurrentCulture) == "unary" && op.Shape[0].SequenceEqual(call.CheckedShape.ToValueArray()));
        }

        if (node is not null)
        {
            return call.With(
                target: new IR.CustomCPU.Unary(unary.UnaryOp, node!.SBP[0], new() { [CostFactorNames.CPUCycles] = node.Cost }, null),
                arguments: new[] { input },
                metadata: call.Metadata);
        }

        return null;
    }
}
