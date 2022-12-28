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
    public override Pattern Pattern { get; } = IsWildcardCall<EndT>("end_call", "end_op",
        IsWildcardCall<T>("mid_call", "mid_op",
            IsWildcardCall<BeginT>("begin_call", "begin_op", IsWildcard("input"))));

    protected virtual Call? GetReplace(Call end_call, IReadOnlyList<Expr> end_call_params,
                             Call mid_call, IReadOnlyList<Expr> mid_call_params,
                             Call begin_call, IReadOnlyList<Expr> begin_call_params,
                             Expr input)
    {
        var new_input = new Var(input.CheckedType!);
        var new_begin_call_params = ReplaceParams(begin_call_params, input, new_input);
        var new_begin_call = begin_call with { Parameters = new(new_begin_call_params) };

        var new_mid_call_params = ReplaceParams(mid_call_params, begin_call, new_begin_call);
        var new_mid_call = mid_call with { Parameters = new(new_mid_call_params) };

        var new_end_call_params = ReplaceParams(end_call_params, mid_call, new_mid_call);
        var new_end_call = end_call with { Parameters = new(new_end_call_params) };

        var fusion = new Call(new Fusion(FullName, ModuleKind, new_end_call, new[] { new_input }), input);
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
    public override Pattern Pattern { get; } = IsWildcardCall<EndT>("end_call", "end_op",
        IsWildcardCall<T>("mid_call", "mid_op",
            IsWildcardCall<BeginT>("begin_lhs_call", "begin_lhs_op", IsWildcard("lhs")),
            IsWildcardCall<BeginT>("begin_rhs_call", "begin_rhs_op", IsWildcard("rhs"))));

    private Call GetReplace(Call end_call, IReadOnlyList<Expr> end_call_params,
                            Call mid_call, IReadOnlyList<Expr> mid_call_params,
                            Call begin_lhs_call, IReadOnlyList<Expr> begin_lhs_call_params,
                            Call begin_rhs_call, IReadOnlyList<Expr> begin_rhs_call_params,
                            Expr lhs, Expr rhs)
    {
        var varIndex = 0;
        var new_args = new List<Var>();
        var new_params = new List<Expr>();
        var new_begin_calls = new List<Expr>();
        if (lhs is not TensorConst)
        {
            var arg = new Var($"input{varIndex++}", lhs.CheckedType!);
            var new_begin_lhs_call_params = ReplaceParams(begin_lhs_call_params, lhs, arg);
            var new_begin_lhs_call = begin_lhs_call with { Parameters = new(new_begin_lhs_call_params) };
            new_params.Add(arg);
            new_params.Add(lhs);
        }

        if (rhs is not TensorConst)
        {
            var arg = new Var($"input{varIndex++}", rhs.CheckedType!);
            var new_begin_rhs_call_params = ReplaceParams(begin_rhs_call_params, rhs, arg);
            var new_begin_rhs_call = begin_rhs_call with { Parameters = new(new_begin_rhs_call_params) };
            new_params.Add(arg);
            new_params.Add(rhs);
        }

        var new_mid_call_params = ReplaceParams(mid_call_params, new List<Expr>() { begin_lhs_call, begin_rhs_call }, new_begin_calls);
        var new_mid_call = mid_call with { Parameters = new(new_mid_call_params) };

        var new_end_call_params = ReplaceParams(end_call_params, mid_call, new_mid_call);
        var new_end_call = end_call with { Parameters = new(new_end_call_params) };

        var fusion = new Call(new Fusion(FullName, ModuleKind, new_end_call, new_args.ToArray()), new_params.ToArray());
        return fusion;
    }
}

[RuleGenerator]
public partial class DataTransferFusion<LoadT, StoreT> : FusionMaker
    where LoadT : Op
    where StoreT : Op
{
    /// <inheritdoc/>
    public override Pattern Pattern { get; } = IsWildcardCall<StoreT>("st_call", "st",
        IsWildcardCall<LoadT>("ld_call", "ld", IsWildcard("input")));

    // replace input with var
    private Call? GetReplace(Call st_call, IReadOnlyList<Expr> st_call_params,
                             Call ld_call, IReadOnlyList<Expr> ld_call_params,
                             Expr input)
    {
        var new_arg = new Var(input.CheckedType!);

        var new_ld_call_params = ReplaceParams(ld_call_params, input, new_arg);
        var new_ld_call = ld_call with { Parameters = new(new_ld_call_params) };

        var new_st_call_params = ReplaceParams(st_call_params, ld_call, new_ld_call);
        var new_st_call = st_call with { Parameters = new(new_st_call_params) };

        var fusion = new Call(new Fusion(FullName, ModuleKind, new_st_call, new[] { new_arg }), input);
        return fusion;
    }
}
