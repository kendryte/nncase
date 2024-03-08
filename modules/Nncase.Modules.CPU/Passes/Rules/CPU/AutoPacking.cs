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

        var newbody = CompilerServices.ERewrite(
            body,
            new IRewriteRule[] {
                new Passes.Rules.CPU.PackSoftmax(),
                new Passes.Rules.CPU.PackSwish(),
                new Passes.Rules.CPU.PackLayerNorm(),
                new Passes.Rules.CPU.PackMatMul(),
                new Passes.Rules.CPU.PackUnary(),
                new Passes.Rules.CPU.PackBinary(),
                new Passes.Rules.CPU.PackTranspose(),
                new Passes.Rules.CPU.PackUnsqueeze(),
                new Passes.Rules.CPU.PackReshape(),
                new Passes.Rules.CPU.PackSlice(),
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
