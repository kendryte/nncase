// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using System.Text;
using System.Text.RegularExpressions;
using System.Threading.Tasks;
using DryIoc.ImTools;
using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.CSharp;
using Microsoft.CodeAnalysis.CSharp.Scripting;
using Microsoft.CodeAnalysis.Scripting;
using NetFabric.Hyperlinq;
using Nncase.CostModel;
using Nncase.IR;
using Nncase.IR.Buffers;
using Nncase.IR.Math;
using Nncase.Passes.Distributed;
using Nncase.PatternMatch;
using Nncase.PatternMatch.F;
using static Nncase.IR.F.Math;
using static Nncase.IR.TypePatternUtility;
using static Nncase.PatternMatch.Utility;

namespace Nncase.Passes.Rules.NTT.CustomOp;

public record ExtraWorkloadGlobals
{
    public Dimension? M;
    public Dimension? K;
    public Dimension? N;
}

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
        IsWildcard("rhs"),
        IsWildcard("scale"));

    private Expr? GetReplace(Call call, MatMul mm, Expr lhs, Expr rhs, Expr scale)
    {
        if (Scheme is null)
        {
            return null;
        }

        // Name pattern
        var node = Scheme.Outputs.FirstOrDefault(op => call.Metadata.OutputNames?[0] is string outputName && Regex.IsMatch(outputName, op.Name ?? string.Empty));

#if false
        if (node is null)
        {
            node = Scheme.Outputs.FirstOrDefault(op =>
                op.Op.ToLower(CultureInfo.CurrentCulture) == "matmul" &&
                op.Shape[0].SequenceEqual(lhs.CheckedShape.ToValueArray()) &&
                op.Shape[1].SequenceEqual(rhs.CheckedShape.ToValueArray()));
        }
#endif

        if (node is not null)
        {
            var extraSize = new long[] { 1 };
            if (!string.IsNullOrEmpty(node.ExtraWorkload))
            {
                var scriptOptions = ScriptOptions.Default
                    .AddReferences(typeof(Dimension).Assembly)
                    .AddImports(typeof(Dimension).Namespace);
                var globals = new ExtraWorkloadGlobals
                {
                    M = lhs.CheckedShape[^2],
                    K = rhs.CheckedShape[^1],
                    N = rhs.CheckedShape[^1],
                };

                var extraDim = CSharpScript.EvaluateAsync<Dimension>(node.ExtraWorkload, scriptOptions, globals: globals).Result;
                extraSize = CompilerServices.GetMaxShape([extraDim]);
            }

            Expr extraWorkload = IR.F.Buffer.Uninitialized(DataTypes.UInt8, TIR.MemoryLocation.Data, extraSize);

            return call.With(
                    target: new IR.CustomNTT.MatMul(null!, null!, false, false, node!.SBP[0], node!.SBP[1], node!.SBP[2], new() { [CostFactorNames.CPUCycles] = node.Cost }, node.CSourcePath, node.FuncName, mm.OutputDataType),
                    arguments: new[] { lhs, rhs, scale, extraWorkload },
                    metadata: call.Metadata);
        }

        return null;
    }
}
