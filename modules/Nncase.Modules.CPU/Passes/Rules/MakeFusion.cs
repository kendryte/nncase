// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Nncase.IR;
using Nncase.IR.CPU;
using Nncase.Passes.Rules.Neutral;
using Nncase.PatternMatch;
using Nncase.Targets;
using static Nncase.PatternMatch.F.Tensors;
using static Nncase.PatternMatch.Utility;
using static Nncase.Utilities.ReplaceUtility;

namespace Nncase.Passes.Rules;

[RuleGenerator]
internal partial class CPUSingleInputFusion<T> : FusionMaker
    where T : Op
{
    public override string ModuleKind { get; } = CPUTarget.Kind;

    public override Pattern Pattern { get; } = IsCallWildcard(
            "call",
            IsOp<T>("op"),
            IsWildcard("input"));

    private Call? GetReplace(Call call, IReadOnlyList<Expr> callParams, Op op, Expr input)
    {
        var newInput = new Var(input.CheckedType!);
        var newCall = ReplaceCallParams(op, callParams, (input, newInput));
        var fusion = new Call(new Fusion(FullName, ModuleKind, newCall, new[] { newInput }), input);
        return fusion;
    }
}

internal sealed class CPUUnaryFusion : CPUSingleInputFusion<CPUUnary>
{
    public override string Name => "Unary";
}
