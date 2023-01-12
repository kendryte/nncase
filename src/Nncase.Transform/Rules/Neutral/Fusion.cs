// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using NetFabric.Hyperlinq;
using Nncase.CodeGen;
using Nncase.IR;
using Nncase.IR.Math;
using Nncase.IR.Tensors;
using Nncase.PatternMatch;
using static Nncase.IR.TypePatternUtility;
using static Nncase.PatternMatch.Utility;
using static Nncase.Utilities.ReplaceUtility;
using ParameterInfo = Nncase.IR.ParameterInfo;
using Tuple = System.Tuple;

namespace Nncase.Transform.Rules.Neutral;

public abstract class FusionMaker : RewriteRule<Pattern>
{
    private int _count;

    public virtual string Name { get; } = "FusionMaker";

    public virtual string ModuleKind { get; } = "StackVM";

    public string FullName => $"{Name}_{_count++}";
}

/// <summary>
/// used for make multi input and multi output fusion.
/// e.g.
///       load    load    load
///         \      |       /
///              LSTM
/// body =      /    \
///        GetItem  GetItem
///           |        |
///         store    store
///
/// body => Call(Target:Fusion(body))
///
/// but something not support
/// 1. not TensorConst
/// 2. Swappable Binary
/// in these cases you should use DoubleInputFusion.
///
/// </summary>
/// <typeparam name="OpT">OpT.</typeparam>
/// <typeparam name="BeginT">Begin process for input.</typeparam>
/// <typeparam name="EndT">End process for output.</typeparam>
/// <typeparam name="DataMaker">Used for set detail pattern.
/// In DataMaker, you should impl your own member
/// which signature is same as "public static (ParameterInfo, Pattern)[] InputsPattern". </typeparam>
[RuleGenerator]
public partial class ComplexFusion<OpT, BeginT, EndT, DataMaker> : FusionMaker
    where OpT : Op
    where BeginT : Op
    where EndT : Op
{
    public static string OutputName = "output";

    public static Pattern ComputePattern { get; } = IsCallWithSpecInput<OpT>("midCall", null!,
        InputsPattern);

    public static (ParameterInfo, Pattern)[] InputsPattern =>
        typeof(DataMaker).GetField("InputsPattern").GetValue(null) as (ParameterInfo, Pattern)[];

    public override Pattern Pattern { get; } = IsAlt(

        // if multi output, then the name by generated should be set null to avoid name conflict
        // if single output, then use the OutputName
        // designed for fusion single output and multi output by only one rule
        MultiOutPattern(ComputePattern, OutputName),
        EndPattern(OutputName));

    /// <summary>
    /// Generate multi output pattern, wrap with GetItem.
    /// </summary>
    /// <param name="inputPattern">Target of GetItem.</param>
    /// <returns>TuplePattern.</returns>
    public static Pattern MultiOutPattern(Pattern inputPattern, string outputName) => IsTuple(
        GenerateRepeatParameters(
            () => IsWildcardCall<EndT>(null!, null!,
                IsWildcardCall<GetItem>(null!, null!, inputPattern))),
        outputName);

    public static Pattern EndPattern(string endCallName) => IsWildcardCall<EndT>(endCallName, null!, ComputePattern);

    /// <summary>
    /// Used for construct wildcard Pattern for inputs from ParameterInfo[].
    /// </summary>
    /// <param name="infos">Parameter Infos.</param>
    /// <typeparam name="BeginT" />
    /// <returns></returns>
    public static (ParameterInfo, Pattern)[] GenerateInputsPattern(params ParameterInfo[] infos) =>
        infos.Select(x => (x, (Pattern)IsWildcardCall<BeginT>(null, null, (string)null))).ToArray();

    /// <summary>
    /// Get input Expr from expr of BeginT.
    /// When you want to modify default behavior, you should override it.
    /// </summary>
    /// <param name="begin"></param>
    /// <returns></returns>
    public virtual Expr GetInputFromBegin(Expr begin)
    {
        return ((Call)begin).Parameters[0];
    }

    /// <summary>
    /// Replace the old expr with new expr for each field.
    /// </summary>
    /// <param name="oldExpr"></param>
    /// <param name="tupleOut"></param>
    /// <param name="newExpr"></param>
    /// <returns></returns>
    public virtual IR.Tuple ReplaceTupleFields(Expr oldExpr, IR.Tuple tupleOut, Expr newExpr)
    {
        var newFields = tupleOut.Fields.Select(end =>
                ReplaceFirst((Call)end, (Expr)ReplaceCallParam(

                    // end is a GetItem(Call(OpT, params))
                    // then end[0] is Call(OpT, params)
                    (Call)((Call)end).Parameters[0],
                    new[] { (oldExpr, newExpr) })))
            .ToArray();
        return tupleOut with { Fields = newFields };
    }

    protected virtual Call? GetReplace(Expr output, Call midCall, IReadOnlyList<Expr> midCallParams)
    {
        int idx = 0;

        // get inputs for spec
        var oldInputs = InputsPattern
            .Select(x => x.Item1.Index)
            .Select(i => midCallParams[i])
            .ToArray();

        // update old input and accumulate new begin and new input
        var (newBegins, newInputs) = oldInputs
            .Aggregate(
                (Array.Empty<Expr>(), Array.Empty<Var>()),
                (tuple, begin) =>
                {
                    var input = GetInputFromBegin(begin);
                    var newInput = new Var($"input_{idx++}", input.CheckedType!);
                    var newBegin = ReplaceCallParam((Call)begin, new[] { (input, (Expr)newInput) });
                    return (
                        tuple.Item1.Append(newBegin).ToArray(),
                        tuple.Item2.Append(newInput).ToArray());
                });

        // update compute
        var newMidCall = ReplaceCallParam(midCall, midCallParams.Zip(newBegins).ToArray());

        // update end
        Expr newBody = output switch
        {
            IR.Tuple tuple => ReplaceTupleFields(midCall, tuple, newMidCall),
            Call endCall => ReplaceCallParam(endCall, new[] { ((Expr)midCall, (Expr)newMidCall) }),
            _ => throw new NotSupportedException("not suppoerted output type"),
        };

        // update parameters
        var parameters = oldInputs.Select(GetInputFromBegin).ToArray();
        var fusion = new Call(new Fusion(FullName, ModuleKind, newBody, newInputs), parameters);
        return fusion;
    }
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

        var fusion = new Call(
            new Fusion(FullName, ModuleKind, new_endCall, new_args.ToArray()),
            newParams.ToArray());
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
