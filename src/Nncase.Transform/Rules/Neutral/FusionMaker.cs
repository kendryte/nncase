// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Collections.Immutable;
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
/// <typeparam name="MidT">OpT.</typeparam>
/// <typeparam name="BeginT">Begin process for input.</typeparam>
/// <typeparam name="EndT">End process for output.</typeparam>
/// In DataMaker, you should impl your own member
/// which signature is same as "public static (ParameterInfo, Pattern)[] InputsPattern". </typeparam>
[RuleGenerator]
public partial class ComplexFusion<MidT, BeginT, EndT> : FusionMaker
    where MidT : Op
    where BeginT : Op
    where EndT : Op
{
    public ComplexFusion()
    {
        Pattern computePattern = IsCallSpecific("midCall", IsOp<MidT>("midCallOp"), InputPatterns.Select(p => (p.Item1, (Pattern)p.Item2)).ToArray());
        Pattern = IsAlt(
          MultiEndPattern("output", computePattern),
          SingleEndPattern("output", computePattern));
    }

    public virtual (ParameterInfo, CallPattern)[] InputPatterns { get; } = Array.Empty<(ParameterInfo, CallPattern)>();

    /// <summary>
    /// Gets if multi output, then the name by generated should be set null to avoid name conflict
    /// if single output, then use the OutputName
    /// designed for fusion single output and multi output by only one rule.
    /// </summary>
    public override Pattern Pattern { get; }

    /// <summary>
    /// useful Util.
    /// </summary>
    /// <param name="infos">infos.</param>
    /// <returns>pair array.</returns>
    public static (ParameterInfo, CallPattern)[] GenerateInputPatterns(params ParameterInfo[] infos) =>
      infos.Select(x => (x, IsCallWildcard(null, IsOp<BeginT>(null, x => true), IsWildcard()))).ToArray();

    /// <summary>
    /// match :
    ///  tuple(
    ///   call(endOp, getitem(input,0),..),
    ///   call(endOp, getitem(input,1),..),
    ///   ..)
    /// </summary>
    /// <param name="endName">end call name.</param>
    /// <param name="inputPattern">input pattern.</param>
    /// <returns>end call pattern.</returns>
    public static Pattern MultiEndPattern(string endName, Pattern inputPattern)
      => PatternMatch.Utility.IsTuple(
          endName,
          IsVArgsRepeat(null, (fields) =>
          {
              return fields.Select((_, i) =>
                IsCallWildcard(endName + $"_{i}", IsOp<EndT>(endName + $"Op_{i}"),
                  IsCallWildcard($"getItem_{i}", IsOp<GetItem>(name: null),
                    inputPattern)))
              .ToArray();
          }));

    /// <summary>
    /// match call(endOp,input).
    /// </summary>
    /// <param name="endCall">end call name.</param>
    /// <param name="inputPattern">input pattern.</param>
    /// <returns>end call pattern.</returns>
    public static Pattern SingleEndPattern(string endCall, Pattern inputPattern) => IsCallWildcard(endCall, IsOp<EndT>(endCall + "Op"), inputPattern);

    protected virtual Call? GetReplace(Call midCall, Op midCallOp, IReadOnlyList<Expr> midCallParams, Expr output, IMatchResult result)
    {
        var newInputs = new List<Var>();
        var newParams = new List<Expr>();

        // 1. update all input call (replace first input by var)
        var midPairs = InputPatterns.Select((p, i) =>
        {
            var beginCallParams = (IReadOnlyList<Expr>)result[p.Item2.Parameters];
            var newVar = new Var($"input_{i}", beginCallParams[0].CheckedType!);
            var newParam = beginCallParams[0];
            newInputs.Add(newVar);
            newParams.Add(newParam);
            var newBeginCall = ReplaceCallParams((Expr)result[p.Item2.Target], beginCallParams, (newParam, newVar));
            return (p.Item1, (Expr)newBeginCall);
        }).ToArray();

        // 2. update mid compute call
        var newMidCall = ReplaceCallParams(midCallOp, midCallParams, midPairs);

        // 3. update output.
        Expr newOutput = output switch
        {
            IR.Tuple endtuple => ReplaceTupleFields(endtuple, midCall, newMidCall, result),
            Call endCall => ReplaceCallParams((Expr)result["outputOp"], (IReadOnlyList<Expr>)result["outputParams"], (midCall, newMidCall)),
            _ => throw new NotSupportedException("not suppoerted output type"),
        };

        var fusion = new Call(new Fusion(FullName, ModuleKind, newOutput, ImmutableArray.CreateRange(newInputs)), ImmutableArray.CreateRange(newParams));
        return fusion;
    }

    private Expr ReplaceTupleFields(IR.Tuple tuple, Call midCall, Call newMidCall, IMatchResult result)
    {
        return new IR.Tuple(ImmutableArray.CreateRange(
          tuple.Fields.Select((_, i) =>
            (Expr)ReplaceCallParams(
                new GetItem(),
                (IReadOnlyList<Expr>)result[$"getItem_{i}Params"],
                (midCall, newMidCall)))));
    }
}

[RuleGenerator]
public partial class SingleInputFusion<T, BeginT, EndT> : FusionMaker
    where T : Op
    where BeginT : Op
    where EndT : Op
{
    /// <inheritdoc/>
    public override Pattern Pattern { get; } = IsCallWildcard("endCall", IsOp<EndT>("endOp"),
        IsCallWildcard("midCall", IsOp<T>("midOp"),
            IsCallWildcard("beginCall", IsOp<BeginT>("beginOp"), IsWildcard("input"))));

    private Call? GetReplace(
        Call endCall, Op endOp, IReadOnlyList<Expr> endCallParams,
        Call midCall, Op midOp, IReadOnlyList<Expr> midCallParams,
        Call beginCall, Op beginOp, IReadOnlyList<Expr> beginCallParams,
        Expr input)
    {
        var newInput = new Var(input.CheckedType!);
        var newBeginCall = ReplaceCallParams(beginOp, beginCallParams, (input, newInput));
        var newMidCall = ReplaceCallParams(midOp, midCallParams, (beginCall, newBeginCall));
        var newEndCall = ReplaceCallParams(endOp, endCallParams, (midCall, newMidCall));

        var fusion = new Call(new Fusion(FullName, ModuleKind, newEndCall, new[] { newInput }), input);
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
    public override Pattern Pattern { get; } = IsCallWildcard("endCall", IsOp<EndT>("endOp"),
        IsCallWildcard("midCall", IsOp<T>("midOp"),
            IsCallWildcard("beginLhsCall", IsOp<BeginT>("beginLhsOp"), IsWildcard("lhs")),
            IsCallWildcard("beginRhsCall", IsOp<BeginT>("beginRhsOp"), IsWildcard("rhs"))));

    private Call GetReplace(Call endCall, Op endOp, IReadOnlyList<Expr> endCallParams,
        Call midCall, Op midOp, IReadOnlyList<Expr> midCallParams,
        Call beginLhsCall, Op beginLhsOp, IReadOnlyList<Expr> beginLhsCallParams,
        Call beginRhsCall, Op beginRhsOp, IReadOnlyList<Expr> beginRhsCallParams,
        Expr lhs, Expr rhs)
    {
        var newArgs = new List<Var>();
        var newParams = new List<Expr>();
        var replace_pairs = new List<(Expr, Expr)>();
        if (lhs is not TensorConst)
        {
            var arg = new Var(lhs.CheckedType!);
            var newBeginLhsCall = ReplaceCallParams(beginLhsOp, beginLhsCallParams, (lhs, arg));
            newArgs.Add(arg);
            newParams.Add(lhs);
            replace_pairs.Add((beginLhsCall, newBeginLhsCall));
        }

        if (rhs is not TensorConst)
        {
            var arg = new Var(rhs.CheckedType!);
            var newBeginRhsCall = ReplaceCallParams(beginRhsOp, beginRhsCallParams, (rhs, arg));
            newArgs.Add(arg);
            newParams.Add(rhs);
            replace_pairs.Add((beginRhsCall, newBeginRhsCall));
        }

        var newMidCall = ReplaceCallParams(midOp, midCallParams, replace_pairs.ToArray());

        var newEndCall = ReplaceCallParams(endOp, endCallParams, (midCall, newMidCall));

        var fusion = new Call(
            new Fusion(FullName, ModuleKind, newEndCall, newArgs.ToArray()),
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
    public override Pattern Pattern { get; } = IsCallWildcard("stCall", IsOp<StoreT>("stOp"),
        IsCallWildcard("ldCall", IsOp<LoadT>("ldOp"), IsWildcard("input")));

    // replace input with var
    private Call? GetReplace(
        Call stCall, Op stOp, IReadOnlyList<Expr> stCallParams,
        Call ldCall, Op ldOp, IReadOnlyList<Expr> ldCallParams,
        Expr input)
    {
        var newArg = new Var(input.CheckedType!);
        var newLdCall = ReplaceCallParams(ldOp, ldCallParams, (input, newArg));
        var newStCall = ReplaceCallParams(stOp, stCallParams, (ldCall, newLdCall));
        var fusion = new Call(new Fusion(FullName, ModuleKind, newStCall, new[] { newArg }), input);
        return fusion;
    }
}
