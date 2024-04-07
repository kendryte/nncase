// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System.Reactive;
using System.Runtime.CompilerServices;
using NetFabric.Hyperlinq;
using Nncase.CodeGen;
using Nncase.IR;
using Nncase.IR.CPU;
using Nncase.IR.Tensors;
using Nncase.PatternMatch;
using Nncase.Targets;
using static Nncase.PatternMatch.Utility;

[assembly: InternalsVisibleTo("Nncase.Tests")]

namespace Nncase.Passes.Rules;

/// <summary>
/// auto distributed the xpu fusion.
/// </summary>
[RuleGenerator]
public sealed partial class AutoPacking : IRewriteRule
{
    public IPattern Pattern { get; } = IsCallWildcard("call", IsFusion("fusion", CPUTarget.Kind, IsWildcard("body"), IsVArgsRepeat("parameters", () => IsVar())));

    private Expr? GetReplace(Call call, Fusion fusion, Expr body, IReadOnlyList<Expr> parameters, IReadOnlyList<Expr> callParams)
    {
        // 1. convert to distribute graph
        if (fusion.Metadata is PackMetaData)
        {
            return null;
        }

        var rank = 1;
        var lane = System.Runtime.Intrinsics.Vector256.IsHardwareAccelerated ? 8 : 4;
        var newbody = CompilerServices.ERewrite(
            body,
            new IRewriteRule[] {
                new Passes.Rules.CPU.PackSoftmax() { Rank = rank, Lane = lane },
                new Passes.Rules.CPU.PackSwish() { Rank = rank, Lane = lane },
                new Passes.Rules.CPU.PackLayerNorm() { Rank = rank, Lane = lane },
                new Passes.Rules.CPU.PackMatMul() { Rank = rank, Lane = lane },
                new Passes.Rules.CPU.PackConv2D() { Rank = rank, Lane = lane },
                new Passes.Rules.CPU.PackUnary() { Rank = rank, Lane = lane },
                new Passes.Rules.CPU.PackBinary() { Rank = rank, Lane = lane },
                new Passes.Rules.CPU.PackTranspose() { Rank = rank, Lane = lane },
                new Passes.Rules.CPU.PackUnsqueeze() { Rank = rank, Lane = lane },
                new Passes.Rules.CPU.PackReshape() { Rank = rank, Lane = lane },
                new Passes.Rules.CPU.PackSlice() { Rank = rank, Lane = lane },
                new Passes.Rules.Neutral.FoldConstCall(),
                new Passes.Rules.CPU.FoldPackUnpack(),
                new Passes.Rules.CPU.FoldPackConcatUnpack(),
            },
            new());

        var newFusion = fusion.With(moduleKind: CPUTarget.Kind, body: newbody, parameters: parameters.Cast<Var>().ToArray());
        newFusion.Metadata = new PackMetaData();
        return new Call(newFusion, callParams.ToArray());
    }

    private sealed class PackMetaData : IR.IRMetadata
    {
    }
}
