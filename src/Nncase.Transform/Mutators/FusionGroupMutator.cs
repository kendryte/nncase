// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.Diagnostics.CodeAnalysis;
using System.IO;
using System.Linq;
using Nncase.IR;
using Nncase.PatternMatch;
using static Nncase.PatternMatch.Utility;


namespace Nncase.Transform.Mutators;

/// <summary>
/// The Fusion Merge Rule.
/// </summary>
public interface IFusionMergeRule
{
    /// <summary>
    /// Gets pattern.
    /// </summary>
    IPattern Pattern { get; }

    /// <summary>
    /// Check the call before the match
    /// </summary>
    /// <param name="usedByAnalysisReslut"></param>
    /// <param name="old_call"></param>
    /// <returns></returns>
    bool CheckMergeBeforeMatchRoot(IUsedByResult usedByAnalysisReslut, Call old_call);

    /// <summary>
    /// Create the pattern.
    /// </summary>
    /// <param name="target_module_kind"></param>
    /// <returns>the pattern.</returns>
    Pattern CreatePattern(string target_module_kind);

    /// <summary>
    /// Get replace expression.
    /// </summary>
    /// <param name="result">Match result.</param>
    /// <param name="options">options.</param>
    /// <returns>Replace expression or null if nothing changed.</returns>
    Expr? GetReplace(Func<Expr, Expr> mergedFusionRewriteCallBack, Func<Fusion, bool> mergedFusionCheckCallBack,
      Func<HashSet<Fusion>, bool> candidateFusionCheckCallBack, Action<HashSet<Fusion>> candidateFusionRecordCallBack,
      IUsedByResult usedByReslut, IMatchResult result, RunPassOptions options);
}

/// <summary>
/// fusion multi input fusion
///   ?      ?       ?
///    \     |      /
///   fusion(...)
///        |
///   z = fusion(y)
/// </summary>
public class MultiInputFusionMergeRule : IFusionMergeRule
{
    /// <summary>
    /// the matched fusion module kind.
    /// </summary>
    public virtual string ModuleKind => Callable.StackVMModuleKind;

    private Pattern? _pattern = null;

    /// <inheritdoc/>
    public IPattern Pattern => _pattern ?? CreatePattern(ModuleKind);

    /// <inheritdoc/>
    public Pattern CreatePattern(string target_module_kind)
    {
        var CalleePattern = IsCall(
          "callee",
          IsFusion("callee_fusion",
            target_module_kind,
            IsWildcard(),
            IsVArgsRepeat(() => IsWildcard())),
          IsVArgsRepeat("callee_inputs", (IReadOnlyList<Expr> callee_inputs) =>
          {
              var pats = new List<Pattern>();
              for (int i = 0; i < callee_inputs.Count; i++)
              {
                  pats.Add(IsWildcard($"callee_input_{i}"));
              }
              return new(pats);
          }));

        var CallerPattern = IsCall(
            "caller",
            IsFusion("caller_fusion",
              target_module_kind,
              IsWildcard(),
              IsVArgs(IsWildcard())),
              CalleePattern
            );
        return CallerPattern;
    }

    private Fusion processMergeSingleInputFusion(Func<Expr, Expr> mergedFusionRewriteCallBack, Call caller, Call callee, Fusion caller_fusion, Fusion callee_fusion)
    {
        // 1. replace the caller_fusion input_var with the callee_fusion body
        var merged_fusion_body = Transform.Mutator.Substitute(
          e => object.ReferenceEquals(e, caller_fusion.Parameters[0]) ?
           callee_fusion.Body :
           null)().Visit(caller_fusion.Body);

        // 2. run call back.
        merged_fusion_body = mergedFusionRewriteCallBack(merged_fusion_body);

        return new Fusion($"{caller_fusion.Name}_{callee_fusion.Name}", ModuleKind, merged_fusion_body, callee_fusion.Parameters);
    }

    /// <inheritdoc/>
    public Expr? GetReplace(Func<Expr, Expr> mergedFusionRewriteCallBack, Func<Fusion, bool> mergedFusionCheckCallBack,
      Func<HashSet<Fusion>, bool> candidateFusionCheckCallBack,
      Action<HashSet<Fusion>> candidateFusionRecordCallBack,
      IUsedByResult usedByReslut, IMatchResult result, RunPassOptions options)
    {
        var caller = (Call)result["caller"];
        var callee = (Call)result["callee"];
        var caller_fusion = (Fusion)result["caller_fusion"];
        var callee_fusion = (Fusion)result["callee_fusion"];
        var callee_inputs = (IReadOnlyList<Expr>)result["callee_inputs"];

        if (usedByReslut.Get(callee).Count > 1)
            return null;

        var candidate_fusions = new HashSet<Fusion>() { caller_fusion, callee_fusion };

        if (!candidateFusionCheckCallBack(candidate_fusions))
            return null;
        // 1. merge new fusion
        var merged_fusion = processMergeSingleInputFusion(mergedFusionRewriteCallBack, caller, callee, caller_fusion, callee_fusion);

        if (mergedFusionCheckCallBack(merged_fusion))
        {
            var new_call = new Call(merged_fusion, ImmutableArray.CreateRange(callee_inputs));
            // 1. transfer the caller usedby info to new_call
            usedByReslut.Transfer(caller, new_call);
            // 2. clear all caller's and callee's usedy info
            usedByReslut.Clear(caller_fusion, caller);
            usedByReslut.Clear(callee, caller);
            usedByReslut.Clear(callee_fusion, callee);
            foreach (var callee_input in callee_inputs)
                usedByReslut.Clear(callee_input, callee);

            // 3. reset the input usedby
            foreach (var callee_input in callee_inputs)
                usedByReslut.Add(callee_input, new_call);
            usedByReslut.Add(merged_fusion, new_call);
            return new_call;
        }
        candidateFusionRecordCallBack(candidate_fusions);
        return null;
    }

    /// <inheritdoc/>
    public bool CheckMergeBeforeMatchRoot(IUsedByResult usedByAnalysisReslut, Call old_call)
    {
        if (usedByAnalysisReslut.Get(old_call).Count > 1)
            return false;
        return true;
    }
}

/// <summary>
/// merge root fusion is multi input.
///  x|fusion(x)  x|fusion(x)   x|fusion(x)
///    \      |        /
///     \     |      / 
///      \   |    /     
///       fusion(...) 
/// </summary>
public class SameInputFusionMergeRule : IFusionMergeRule
{
    /// <summary>
    /// Get ModuleKind.
    /// </summary>
    public virtual string ModuleKind => Callable.StackVMModuleKind;

    private Pattern? _pattern = null;

    /// <inheritdoc/>
    public Pattern CreatePattern(string target_module_kind)
    {
        var inputPat = IsWildcard("input");

        var CallerPattern = IsCall(
            "caller",
            IsFusion("caller_fusion",
              target_module_kind,
              IsWildcard(),
              IsVArgsRepeat(() => IsWildcard())),
            IsVArgsRepeat("caller_inputs", (IReadOnlyList<Expr> caller_inputs) =>
            {
                var callee_patterns = new List<Pattern>();
                for (int i = 0; i < caller_inputs.Count; i++)
                {
                    /* 
                    note the OrPattern need care the order.
                    pat  = or(input,call(input))  
                    expr = (call(x), x)
                    the input will first match to `call(x)`, then can't match the x.
                    */
                    callee_patterns.Add(
                      IsAlt(
                        IsCall($"callee_{i}",
                          IsFusion($"callee_fusion_{i}", target_module_kind, IsWildcard(), IsVArgs(IsWildcard())),
                          inputPat),
                        inputPat
                      )
                    );

                }
                return new(callee_patterns);
            })
        );
        return CallerPattern;
    }

    /// <inheritdoc/>
    public IPattern Pattern => _pattern ?? CreatePattern(ModuleKind);


    /// <inheritdoc/>
    public bool CheckMergeBeforeMatchRoot(IUsedByResult usedByAnalysisReslut, Call old_call)
    {
        return true;
    }

    bool processFusionMerge(Func<Expr, Expr> mergedFusionRewriteCallBack, Func<HashSet<Fusion>, bool> candidate_fusion_checker, Expr caller, Fusion caller_fusion,
      IReadOnlyList<Expr> caller_inputs, Expr input, IMatchResult result, out HashSet<Fusion> candidate_fusions, out Fusion merged_fusion)
    {
        merged_fusion = null!;
        candidate_fusions = new() { caller_fusion };
        var calleeBodyMap = new Dictionary<Var, Expr>(ReferenceEqualityComparer.Instance);
        var multiVarMap = new Dictionary<Var, Var>();
        var new_fusion_input_var = new Var(input.CheckedType!);
        string new_fusion_name = $"{caller_fusion.Name}";

        bool has_fusion_for_merge = false; // because of it can match fusion(var)

        for (int i = 0; i < caller_inputs.Count; i++)
        {
            if (caller_inputs[i] is Call { Target: Fusion })
            {
                Fusion callee_fusion;
                try
                {
                    callee_fusion = (Fusion)result[$"callee_fusion_{i}"];
                }
                catch (KeyNotFoundException)
                {
                    // when matched fusion(fusion(x,y)), the input => fusion(x,y) 
                    return false;
                }
                calleeBodyMap.Add(caller_fusion.Parameters[i], callee_fusion.Body);
                new_fusion_name += ("_" + callee_fusion.Name);
                candidate_fusions.Add(callee_fusion);
                multiVarMap.Add(callee_fusion.Parameters[0], new_fusion_input_var);
                has_fusion_for_merge = true;
            }
            else
            {
                multiVarMap.Add(caller_fusion.Parameters[i], new_fusion_input_var);
            }
        }

        if (!has_fusion_for_merge)
            return false;

        if (candidate_fusion_checker(candidate_fusions))
            return false;

        // 1. replace the caller_fusion input_var with the callee_fusion_i body
        Expr merged_fusion_body = caller_fusion.Body;
        if (calleeBodyMap.Count > 0)
        {
            merged_fusion_body = Transform.Mutator.Substitute(e =>
            {
                if (e is Var ve && calleeBodyMap.TryGetValue(ve, out var new_e))
                    return new_e;
                return null;
            })().Visit(merged_fusion_body);
        }
        // 2. replace the all input var to new input var
        merged_fusion_body = Transform.Mutator.Substitute(e =>
        {
            if (e is Var ve && multiVarMap.TryGetValue(ve, out var new_e))
                return new_e;
            return null;
        })().Visit(merged_fusion_body);


        // 2. run call back.
        merged_fusion_body = mergedFusionRewriteCallBack(merged_fusion_body);

        merged_fusion = new Fusion(new_fusion_name, ModuleKind, merged_fusion_body, new[] { new_fusion_input_var });

        return true;
    }

    /// <inheritdoc/>
    public Expr? GetReplace(
      Func<Expr, Expr> mergedFusionRewriteCallBack,
      Func<Fusion, bool> mergedFusionCheckCallBack,
      Func<HashSet<Fusion>, bool> candidateFusionCheckCallBack,
      Action<HashSet<Fusion>> candidateFusionRecordCallBack,
      IUsedByResult usedByReslut, IMatchResult result, RunPassOptions options)
    {
        var caller = (Expr)result["caller"];
        var caller_fusion = (Fusion)result["caller_fusion"];
        var caller_inputs = (IReadOnlyList<Expr>)result["caller_inputs"];
        var input = (Expr)result["input"];

        // it will match  fusion2(fusion1(x,y)), then  input => fusion1(x,y), for merge cycle first so skip it.
        // if (input is Call { Target: Fusion input_fusion } && input_fusion.Parameters.Count != 1)
        //     return false;

        if (caller_inputs.Count == 1)
        {
            /* when match, skip.
                x
                |
              f1(x)
             */
            if (caller_inputs[0] is not Call { Target: Fusion candidate_fusion })
                return false;
            /* skip.
              v0 = f0(x,y,z)
                  |
              v1 = f1(v0)
            */
            if (candidate_fusion.Parameters.Count > 1)
                return false;
            /* skip when it have other user.
                      v0 = f0(x)
                    /          |
            v2 =f2(v0)      v1 = f1(v0)
            */
            // todo. now can't find the mini case.
            if (usedByReslut.Get(caller_inputs[0]).Count > 1)
                return false;
        }

        /*  1. when cycle like:
            x                x                  x 
           /  \           /  |  \         /  |  \      \
       f1(x)  f2(x)     | f1(x) f2(x)   |   |   f1(x) f2(x)
           \  /         \   |   /       \   |   /      /
          f3(x,y)       f3(x,y,z)        f3(x,x,y,z) 
        */
        if (caller_inputs.Count > 1)
        {
            var input_users = new HashSet<Expr>(usedByReslut.Get(input), ReferenceEqualityComparer.Instance);
            // 1. remove the all mid fusion users.
            foreach (var caller_input in caller_inputs)
            {
                if (!object.ReferenceEquals(caller_input, input))
                {
                    if (!input_users.Remove(caller_input))
                        return false;
                }
            }
            // 2. if have more than one users, return false.
            if (input_users.Count > 1)
                return false;
            // 3. final user must be caller.
            if (input_users.Count == 1 && !input_users.Remove(caller))
                return false;
        }

        if (!processFusionMerge(mergedFusionRewriteCallBack, candidateFusionCheckCallBack, caller, caller_fusion, caller_inputs,
           input, result,
            out var candidate_fusions, out var merged_fusion))
            return null;

        if (mergedFusionCheckCallBack(merged_fusion))
        {
            var new_call = new Call(merged_fusion, input);
            // 1. transfer the caller usedby info to new_call
            usedByReslut.Transfer(caller, new_call);
            // 2. clear all caller's and callee's usedy info
            usedByReslut.Clear(caller_fusion, caller);
            for (int i = 0; i < caller_inputs.Count; i++)
            {
                usedByReslut.Clear(caller_inputs[i], caller);
                if (caller_inputs[i] is Call { Target: Fusion callee_fusion })
                    usedByReslut.Clear(callee_fusion, caller_inputs[i]);
                if (!object.ReferenceEquals(caller_inputs[i], input))
                    usedByReslut.Clear(input, caller_inputs[i]);
            }

            // 3. reset the input usedby
            usedByReslut.Add(input, new_call);
            usedByReslut.Add(merged_fusion, new_call);
            return new_call;
        }
        else
            candidateFusionRecordCallBack(candidate_fusions);
        return null;
    }
}


/// <summary>
/// the fusion group mutator.
/// </summary>
public class FusionGroupMutator : ExprMutator
{
    sealed class FusionMergeCandidateComparer : IEqualityComparer<HashSet<Fusion>>
    {
        public bool Equals(HashSet<Fusion>? x, HashSet<Fusion>? y) => (x, y) switch
        {
            (null, null) => true,
            (null, _) => false,
            (_, null) => false,
            (var lhs, var rhs) => GetHashCode(lhs) == GetHashCode(rhs)
        };

        public int GetHashCode([DisallowNull] HashSet<Fusion> obj)
        {
            HashCode hash = new();
            foreach (var o in obj)
            {
                hash.Add(ReferenceEqualityComparer.Instance.GetHashCode(obj));
            }
            return hash.ToHashCode();
        }
    }

    /// <summary>
    /// Get the run pass options.
    /// </summary>
    public readonly RunPassOptions PassOptions;
    
    private readonly IUsedByResult _usedByReslut;
    
    /// <summary>
    /// Get the Pre Order Rules.
    /// </summary>
    public readonly IReadOnlyList<IFusionMergeRule> PreOrderMergeRules;

    /// <summary>
    /// Get the Post Order Rules.
    /// </summary>
    public readonly IReadOnlyList<IFusionMergeRule> PostOrderMergeRules;

    /// <summary>
    /// cache the check result.
    /// </summary>
    private readonly Dictionary<HashSet<Fusion>, bool> _candidateFusionCache;

    /// <summary>
    /// Get the merge check cache result.
    /// </summary>
    public IReadOnlyDictionary<HashSet<Fusion>, bool> FusionMergeCandidateCache => _candidateFusionCache;

    /// <summary>
    /// ctor.
    /// </summary>
    /// <param name="usedByAnalysisReslut">the usedby analysis.</param>
    /// <param name="preOrderfusionRules">pre order rules. </param>
    /// <param name="postOrderfusionRules">post order rules.</param>
    /// <param name="passOptions">pass options. </param>
    public FusionGroupMutator(IUsedByResult usedByAnalysisReslut,
      IEnumerable<IFusionMergeRule> preOrderfusionRules,
      IEnumerable<IFusionMergeRule> postOrderfusionRules, RunPassOptions passOptions)
    {
        _usedByReslut = usedByAnalysisReslut;
        PreOrderMergeRules = preOrderfusionRules.ToList();
        PostOrderMergeRules = postOrderfusionRules.ToList();
        PassOptions = passOptions;
        _candidateFusionCache = new(new FusionMergeCandidateComparer());
    }

    /// <summary>
    /// update merged fusion call.
    /// </summary>
    /// <param name="merge_call"></param>
    /// <returns></returns>
    public virtual Call MergedFusionCallPreOrderCallBack(Call merge_call)
    {
        return merge_call with { Parameters = new(merge_call.Parameters.Select(Visit)) };
    }

    /// <summary>
    /// update the merged fusion post order
    /// </summary>
    /// <param name="call"></param>
    /// <returns></returns>
    public virtual Call MergedFusionCallPostOrderCallBack(Call call)
    {
        return call;
    }

    /// <summary>
    /// check the merged fusion is valid.
    /// </summary>
    /// <param name="merged_fusion">merged fusion.</param>
    /// <returns>bool.</returns>
    public virtual bool MergedFusionCheckCallBack(Fusion merged_fusion)
    {
        return true;
    }

    private bool candidateFusionCheckCallBack(HashSet<Fusion> candidateFusions)
    {
        if (candidateFusions.Count <= 1)
            throw new InvalidDataException("The candidates less than 2!");
        if (!_candidateFusionCache.TryGetValue(candidateFusions, out var ret))
            return true;
        if (ret != false)
            throw new InvalidDataException("Only cache failed candidates!");
        return false;
    }

    private void candidateFusionRecordCallBack(HashSet<Fusion> candidateFusions)
    {
        if (candidateFusions.Count <= 1)
            throw new InvalidDataException("The candidates less than 2!");
        _candidateFusionCache.Add(candidateFusions, false);
    }

    /// <summary>
    /// when fusion merged, maybe need rewrite somethings.
    /// </summary>
    /// <param name="merged_fusion_body">merged fusion body.</param>
    /// <returns>rewrited body.</returns>
    public virtual Expr MergedFusionRewriteCallBack(Expr merged_fusion_body)
    {
        return merged_fusion_body;
    }

    /// <summary>
    /// try merge fusion from the old call.
    /// </summary>
    /// <param name="rules">rules.</param>
    /// <param name="old_call">current call.</param>
    /// <param name="new_call">returned new call.</param>
    /// <returns>merged status. </returns>
    public bool tryMergeFusion(IReadOnlyList<IFusionMergeRule> rules, Call old_call, out Call new_call)
    {
        new_call = null!;
        foreach (var rule in rules)
        {
            if (!rule.CheckMergeBeforeMatchRoot(_usedByReslut, old_call))
                continue;

            if (!CompilerServices.TryMatchRoot(old_call, rule.Pattern, new() { RewriteMemo = ExpressionMemo }, out var result))
                continue;

            if (rule.GetReplace(
              MergedFusionRewriteCallBack, MergedFusionCheckCallBack,
              candidateFusionCheckCallBack, candidateFusionRecordCallBack,
              _usedByReslut, result, PassOptions) is Call replaced_call)
            {
                new_call = replaced_call;
                return true;
            }
        }

        return false;
    }

    /// <inheritdoc/>
    public override Expr Visit(Fusion expr) => expr;

    /// <inheritdoc/>
    public override Expr Visit(Call expr)
    {
        if (!ExpressionMemo.TryGetValue(expr, out var result))
        {
            if (expr is Call { Target: Fusion })
            {
                var last_call = expr;
                Call merged_call = null!;
                while (tryMergeFusion(PreOrderMergeRules, last_call, out merged_call))
                    last_call = merged_call;

                if (!object.ReferenceEquals(last_call, expr))
                {
                    IsMutated = true;
                    var new_call = MergedFusionCallPreOrderCallBack(last_call);
                    updateCallUsedBy(last_call, new_call);
                    ExpressionMemo.Add(expr, new_call);
                    return new_call;
                }
            }

            Visit(expr.Target);
            foreach (var param in expr.Parameters)
            {
                Visit(param);
            }
            result = VisitLeaf(expr);
            ExpressionMemo.Add(expr, result);
        }
        return result;
    }


    /// <inheritdoc/>
    public override Expr MutateLeaf(Call expr)
    {
        if (expr is Call { Target: Fusion })
        {
            if (ExpressionMemo.TryGetValue(expr, out var result))
                return result;

            var last_call = expr;
            Call merged_call = null!;
            while (tryMergeFusion(PostOrderMergeRules, last_call, out merged_call))
                last_call = merged_call;
            var new_call = MergedFusionCallPostOrderCallBack(last_call);
            if (!object.ReferenceEquals(new_call, last_call))
                updateCallUsedBy(last_call, new_call);
            return last_call;
        }
        return expr;
    }

    /// <inheritdoc/>
    public override Expr VisitLeaf(Call expr)
    {
        var nexpr = MutateLeaf(expr);
        if (!object.ReferenceEquals(expr, nexpr))
        {
            IsMutated = true;
            return nexpr;
        }

        if (!IsMutated)
        {
            return expr;
        }

        var with_new = expr with
        {
            Target = Visit(expr.Target),
            Parameters = MutateArray(expr.Parameters, Visit),
        };
        updateCallUsedBy(expr, with_new);
        return with_new;
    }

    private void updateCallUsedBy(Call old_call, Call new_call)
    {
        /* update the usedy info */
        // 1. transfer the caller usedby info to new_call
        _usedByReslut.Transfer(old_call, new_call);
        // 2. clear all caller's and callee's usedy info
        _usedByReslut.Clear(old_call.Target, old_call);
        foreach (var param in old_call.Parameters)
            _usedByReslut.Clear(param, old_call);

        // 3. reset the input usedby
        _usedByReslut.Add(new_call.Target, new_call);
        foreach (var param in new_call.Parameters)
            _usedByReslut.Add(param, new_call);
    }

}
