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
using NetFabric.Hyperlinq;
using Nncase.CostModel;
using Nncase.IR;
using Nncase.IR.CustomNTT;
using Nncase.IR.Math;
using Nncase.IR.Tensors;
using Nncase.Passes.Distributed;
using Nncase.PatternMatch;
using Nncase.PatternMatch.F;
using static Nncase.IR.F.Math;
using static Nncase.IR.TypePatternUtility;
using static Nncase.PatternMatch.F.Math;
using static Nncase.PatternMatch.F.NN;
using static Nncase.PatternMatch.F.Tensors;
using static Nncase.PatternMatch.Utility;

namespace Nncase.Passes.Rules.NTT.CustomOp;

[RuleGenerator]
public partial class ToCustomLayerNorm : RewriteRule<Pattern>
{
    public ToCustomLayerNorm(CustomOpScheme scheme, bool withCast)
    {
        if (withCast)
        {
            Pattern = IsCast(
                "cast",
                "castCall",
                _ => true,
                IsAlt(GetLayerNorm(), GetLayernromWithScale()));
        }
        else
        {
            Pattern = GetLayerNorm();
        }

        Scheme = scheme;
    }

    public CustomOpScheme Scheme { get; }

    /// <inheritdoc/>
    public override Pattern Pattern { get; }

    private Pattern GetLayerNorm() => PatternMatch.F.NN.IsLayerNorm(
        "ln",
        "lnCall",
        _ => true,
        IsWildcard("input"),
        IsWildcard("scale"),
        IsWildcard("bias"));

    private Pattern GetLayernromWithScale() => Nncase.PatternMatch.F.Math.IsBinary(
        "binary",
        "binaryCall",
        _ => true,
        PatternMatch.F.NN.IsLayerNorm(
            "ln",
            "lnCall",
            _ => true,
            IsWildcard("input"),
            IsWildcard("scale"),
            IsWildcard("bias")),
        IsTensorConst("postScale") with { TypePattern = IsScalar() | HasShape(new RankedShape(1)) });

    private Expr? GetReplace(IMatchResult result)
    {
        var lnCall = (Call)result["lnCall"];
        var ln = (IR.NN.LayerNorm)lnCall.Target;
        var input = (Expr)result["input"];
        var scale = (Expr)result["scale"];
        var bias = (Expr)result["bias"];

        if (Scheme is null)
        {
            return null;
        }

        // Name pattern
        var node = Scheme.Outputs.FirstOrDefault(op => lnCall.Metadata.OutputNames?[0] is string outputName && Regex.IsMatch(outputName, op.Name ?? string.Empty));

        var newType = lnCall.CheckedDataType;
        try
        {
            var cast = (Cast)result["cast"];
            newType = cast.NewType;
        }
        catch
        {
            // do nothing
        }

        if (node is not null)
        {
            try
            {
                var postScale = (TensorConst)result["postScale"];
                var newPostScale = postScale.Value.Reshape([]);
                return lnCall.With(
                    target: new IR.CustomNTT.LayerNorm(ln.Axis, ln.Epsilon, ln.UseMean, ln.ChannelFirst, [], node!.SBP[0], node!.SBP[1], node!.SBP[2], node!.SBP[3], new() { [CostFactorNames.CPUCycles] = node.Cost }, node.CSourcePath, node.FuncName, newType),
                    arguments: new[] { input, scale, bias, newPostScale },
                    metadata: lnCall.Metadata);
            }
            catch (KeyNotFoundException)
            {
                var postScale = Tensor.FromScalar(DataTypes.Float32, 1f).CastTo(scale.CheckedDataType);
                return lnCall.With(
                    target: new IR.CustomNTT.LayerNorm(ln.Axis, ln.Epsilon, ln.UseMean, ln.ChannelFirst, [], node!.SBP[0], node!.SBP[1], node!.SBP[2], node!.SBP[3], new() { [CostFactorNames.CPUCycles] = node.Cost }, node.CSourcePath, node.FuncName, newType),
                    arguments: new[] { input, scale, bias, postScale },
                    metadata: lnCall.Metadata);
            }
        }

        return null;
    }
}
