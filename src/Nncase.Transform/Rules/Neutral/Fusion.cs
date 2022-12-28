// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using Nncase.CodeGen;
using Nncase.IR;
using Nncase.PatternMatch;
using static Nncase.PatternMatch.Utility;
using static Nncase.Utilities.ReplaceUtility;

namespace Nncase.Transform.Rules.Neutral;

public abstract class FusionMaker : RewriteRule<Pattern>
{
    private int _count;

    public virtual string Name { get; } = "FusionMaker";

    public virtual string ModuleKind { get; } = "StackVM";

    public string FullName => $"{Name}_{_count++}";
}

[RuleGenerator]
public partial class SingleInputFusion<T, BeginT, EndT> : FusionMaker
    where T : Op
    where BeginT : Op
    where EndT : Op
{
    /// <inheritdoc/>
    public override Pattern Pattern { get; } = IsWildcardCall<EndT>("endCall", null!,
        IsWildcardCall<T>("midCall", null!,
            IsWildcardCall<BeginT>("beginCall", null!, IsWildcard("input"))));

    protected virtual Call? GetReplace(Call endCall, IReadOnlyList<Expr> endCallParams,
                             Call midCall, IReadOnlyList<Expr> midCallParams,
                             Call beginCall, IReadOnlyList<Expr> beginCallParams,
                             Expr input)
    {
        var new_input = new Var(input.CheckedType!);
        var new_beginCallParams = ReplaceParams(beginCallParams, input, new_input);
        var new_beginCall = beginCall with { Parameters = new(new_beginCallParams) };

        var new_midCallParams = ReplaceParams(midCallParams, beginCall, new_beginCall);
        var new_midCall = midCall with { Parameters = new(new_midCallParams) };

        var new_endCallParams = ReplaceParams(endCallParams, midCall, new_midCall);
        var new_endCall = endCall with { Parameters = new(new_endCallParams) };

        var fusion = new Call(new Fusion(FullName, ModuleKind, new_endCall, new[] { new_input }), input);
        return fusion;
    }
}

[RuleGenerator]
public partial class DoubleInputFusion<T, BeginT, EndT> : FusionMaker
    where T : Op
    where BeginT : Op
    where EndT : Op
{
    /// <inheritdoc/>
    public override Pattern Pattern { get; } = IsWildcardCall<EndT>("endCall", null!,
        IsWildcardCall<T>("midCall", null!,
            IsWildcardCall<BeginT>("beginLhsCall", null!, IsWildcard("lhs")),
            IsWildcardCall<BeginT>("beginRhsCall", null!, IsWildcard("rhs"))));

    private Call GetReplace(Call endCall, IReadOnlyList<Expr> endCallParams,
                            Call midCall, IReadOnlyList<Expr> midCallParams,
                            Call beginLhsCall, IReadOnlyList<Expr> beginLhsCallParams,
                            Call beginRhsCall, IReadOnlyList<Expr> beginRhsCallParams,
                            Expr lhs, Expr rhs)
    {
        var new_args = new List<Var>();
        var newParams = new List<Expr>();
        var replace_pairs = new List<(Expr, Expr)>();
        if (lhs is not TensorConst)
        {
            var arg = new Var(lhs.CheckedType!);
            var new_beginLhsCallParams = ReplaceParams(beginLhsCallParams, lhs, arg);
            var new_beginLhsCall = beginLhsCall with { Parameters = new(new_beginLhsCallParams) };
            new_args.Add(arg);
            newParams.Add(lhs);
            replace_pairs.Add((beginLhsCall, new_beginLhsCall));
        }

        if (rhs is not TensorConst)
        {
            var arg = new Var(rhs.CheckedType!);
            var new_beginRhsCallParams = ReplaceParams(beginRhsCallParams, rhs, arg);
            var new_beginRhsCall = beginRhsCall with { Parameters = new(new_beginRhsCallParams) };
            new_args.Add(arg);
            newParams.Add(rhs);
            replace_pairs.Add((beginRhsCall, new_beginRhsCall));
        }

        var new_midCallParams = ReplaceParams(midCallParams, replace_pairs);
        var new_midCall = midCall with { Parameters = new(new_midCallParams) };

        var new_endCallParams = ReplaceParams(endCallParams, midCall, new_midCall);
        var new_endCall = endCall with { Parameters = new(new_endCallParams) };

        var fusion = new Call(new Fusion(FullName, ModuleKind, new_endCall, new_args.ToArray()), newParams.ToArray());
        return fusion;
    }
}

[RuleGenerator]
public partial class DataTransferFusion<LoadT, StoreT> : FusionMaker
    where LoadT : Op
    where StoreT : Op
{
    /// <inheritdoc/>
    public override Pattern Pattern { get; } = IsWildcardCall<StoreT>("stCall", null!,
        IsWildcardCall<LoadT>("ldCall", null!, IsWildcard("input")));

    // replace input with var
    private Call? GetReplace(Call stCall, IReadOnlyList<Expr> stCallParams,
                             Call ldCall, IReadOnlyList<Expr> ldCallParams,
                             Expr input)
    {
        var new_arg = new Var(input.CheckedType!);

        var new_ldCallParams = ReplaceParams(ldCallParams, input, new_arg);
        var new_ldCall = ldCall with { Parameters = new(new_ldCallParams) };

        var new_stCallParams = ReplaceParams(stCallParams, ldCall, new_ldCall);
        var new_stCall = stCall with { Parameters = new(new_stCallParams) };

        var fusion = new Call(new Fusion(FullName, ModuleKind, new_stCall, new[] { new_arg }), input);
        return fusion;
    }
}
