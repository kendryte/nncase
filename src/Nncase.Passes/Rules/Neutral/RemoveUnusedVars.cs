// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.Linq;
using System.Reactive;
using NetFabric.Hyperlinq;
using Nncase.IR;
using Nncase.Passes;
using Nncase.PatternMatch;
using static Nncase.IR.F.NN;
using static Nncase.IR.F.Tensors;
using static Nncase.IR.TypePatternUtility;
using static Nncase.PatternMatch.F.Math;
using static Nncase.PatternMatch.Utility;

namespace Nncase.Passes.Rules.Neutral;

[RuleGenerator]
public sealed partial class RemoveUnusedVarsByCall : IRewriteRule
{
    /// <inheritdoc/>
    public IPattern Pattern { get; } = IsCall(
        "call",
        IsFunction("function", IsWildcard("body"), IsVArgsRepeat("vparams", IsWildcard)),
        IsVArgsRepeat("vargs", IsWildcard));

    private Expr? GetReplace(Call call, Function function, Expr body)
    {
        int unusedVars = 0;
        var usedVars = new List<int>();
        for (int i = 0; i < function.Parameters.Length; i++)
        {
            var var = function.Parameters[i];
            if (var.Users.Count() == 1)
            {
                unusedVars++;
            }
            else
            {
                usedVars.Add(i);
            }
        }

        if (unusedVars != 0)
        {
            var newVarsMap = new Dictionary<Var, Var>(ReferenceEqualityComparer.Instance);
            var newVars = new List<Var>();
            var newArgs = new List<Expr>();
            foreach (var i in usedVars)
            {
                var var = function.Parameters[i];
                var callArg = call.Arguments[i];
                var newVar = var.With();
                newVars.Add(newVar);
                newVarsMap.Add(var, newVar);
                newArgs.Add(callArg);
            }

            var cloner = new VarReplacer(newVarsMap);
            var newBody = cloner.Clone(body, default);
            var newFunc = function.With(body: newBody, parameters: newVars.ToArray());
            return call.With(newFunc, newArgs.ToArray());
        }

        return null;
    }
}

[RuleGenerator]
public sealed partial class RemoveUnusedVarsByIf : IRewriteRule
{
    /// <inheritdoc/>
    public IPattern Pattern { get; } = IsIf(
        "call",
        IsFunction("thenFunc", IsWildcard("thenBody"), IsVArgsRepeat("thenParams", IsWildcard)),
        IsFunction("elseFunc", IsWildcard("elseBody"), IsVArgsRepeat("elseParams", IsWildcard)),
        IsVArgsRepeat("vargs", IsWildcard));

    private Expr? GetReplace(If call, Function thenFunc, Function elseFunc, Expr thenBody, Expr elseBody)
    {
        int unusedVars = 0;
        var usedVars = new List<int>();
        for (int i = 0; i < thenFunc.Parameters.Length; i++)
        {
            var thenVar = thenFunc.Parameters[i];
            var elseVar = elseFunc.Parameters[i];
            if (thenVar.Users.Count() == 1
                && elseVar.Users.Count() == 1)
            {
                unusedVars++;
            }
            else
            {
                usedVars.Add(i);
            }
        }

        if (unusedVars != 0)
        {
            var newVarsMap = new Dictionary<Var, Var>(ReferenceEqualityComparer.Instance);
            var newThenVars = new List<Var>();
            var newElseVars = new List<Var>();
            var newArgs = new List<Expr>();
            foreach (var i in usedVars)
            {
                var thenVar = thenFunc.Parameters[i];
                var elseVar = elseFunc.Parameters[i];
                var callArg = call.Arguments[i];
                var newThenVar = thenVar.With();
                var newElseVar = elseVar.With();
                newThenVars.Add(newThenVar);
                newElseVars.Add(newElseVar);
                newVarsMap.Add(thenVar, newThenVar);
                newVarsMap.Add(elseVar, newElseVar);
                newArgs.Add(callArg);
            }

            var cloner = new VarReplacer(newVarsMap);
            var newThenBody = cloner.Clone(thenBody, default);
            var newElseBody = cloner.Clone(elseBody, default);
            var newThen = thenFunc.With(body: newThenBody, parameters: newThenVars.ToArray());
            var newElse = elseFunc.With(body: newElseBody, parameters: newElseVars.ToArray());
            return call.With(then: newThen, @else: newElse, arguments: newArgs.ToArray());
        }

        return null;
    }
}

internal sealed class VarReplacer : ExprCloner<Unit>
{
    private readonly Dictionary<Var, Var> _newVars;

    public VarReplacer(Dictionary<Var, Var> newVars)
    {
        _newVars = newVars;
    }

    protected override Expr VisitVar(Var var, Unit state)
    {
        return _newVars[var];
    }
}
