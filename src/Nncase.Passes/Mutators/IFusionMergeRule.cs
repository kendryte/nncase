// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.Diagnostics.CodeAnalysis;
using System.IO;
using System.Linq;
using System.Reactive;
using NetFabric.Hyperlinq;
using Nncase.IR;
using Nncase.Passes.Analysis;
using Nncase.PatternMatch;
using static Nncase.PatternMatch.Utility;

namespace Nncase.Passes.Mutators;

/// <summary>
/// The Fusion Merge Pass.
/// </summary>
public interface IMergeRewriteRule
{
    /// <summary>
    /// Gets pattern.
    /// </summary>
    IPattern Pattern { get; }

    /// <summary>
    /// Create the pattern.
    /// </summary>
    /// <returns>the pattern.</returns>
    Pattern CreatePattern(string target_module_kind);

    /// <summary>
    /// Get replace expression.
    /// </summary>
    /// <returns>Replace expression or null if nothing changed.</returns>
    Expr? GetReplace(
      Func<Expr, Expr> mergedFusionRewriteCallBack,
      Func<Fusion, HashSet<Fusion>, bool> mergedFusionCheckCallBack,
      Func<HashSet<Fusion>, bool> candidateFusionCheckCallBack,
      Action<HashSet<Fusion>> candidateFusionRecordCallBack,
      IExprUserAnalysisResult usedByReslut,
      IMatchResult result,
      RunPassContext options);

    protected sealed class FusionMerger : ExprCloner<Unit>
    {
        private readonly IReadOnlyDictionary<Var, Expr> _calleeBodyMap;
        private readonly IReadOnlyDictionary<Var, Var> _multiVarMap;

        public FusionMerger(IReadOnlyDictionary<Var, Expr> calleeBodyMap, IReadOnlyDictionary<Var, Var> multiVarMap)
        {
            _calleeBodyMap = calleeBodyMap;
            _multiVarMap = multiVarMap;
        }

        protected override Expr VisitLeafVar(Var expr, Unit context)
        {
            if (_calleeBodyMap.TryGetValue(expr, out var callee))
            {
                return Visit(callee, context);
            }

            if (_multiVarMap.TryGetValue(expr, out var newVar))
            {
                return newVar;
            }

            throw new InvalidOperationException();
        }
    }
}

/// <summary>
/// fusion multi input fusion
///   ?      ?       ?
///    \     |      /
///   fusion(...)
///        |
///   z = fusion(y).
/// </summary>
public class MultiInputFusionMergeRule : IMergeRewriteRule
{
    private Pattern? _pattern;

    /// <summary>
    /// Gets the matched fusion module kind.
    /// </summary>
    public virtual string ModuleKind => Callable.StackVMModuleKind;

    /// <inheritdoc/>
    public IPattern Pattern => _pattern ??= CreatePattern(ModuleKind);

    /// <inheritdoc/>
    public virtual Pattern CreatePattern(string target_module_kind)
    {
        var calleePattern = IsCall(
          "callee",
          IsFusion(
              "callee_fusion",
              target_module_kind,
              IsWildcard(),
              IsVArgsRepeat(() => IsWildcard())),
          IsVArgsRepeat("callee_inputs", (ReadOnlySpan<Expr> callee_inputs) =>
          {
              var pats = new List<Pattern>();
              for (int i = 0; i < callee_inputs.Length; i++)
              {
                  pats.Add(IsWildcard($"callee_input_{i}"));
              }

              return pats;
          }));

        var callerPattern = IsCall(
            "caller",
            IsFusion(
                "caller_fusion",
                target_module_kind,
                IsWildcard(),
                IsVArgs(IsWildcard())),
            calleePattern);
        return callerPattern;
    }

    /// <inheritdoc/>
    public Expr? GetReplace(
      Func<Expr, Expr> mergedFusionRewriteCallBack,
      Func<Fusion, HashSet<Fusion>, bool> mergedFusionCheckCallBack,
      Func<HashSet<Fusion>, bool> candidateFusionCheckCallBack,
      Action<HashSet<Fusion>> candidateFusionRecordCallBack,
      IExprUserAnalysisResult usedByReslut,
      IMatchResult result,
      RunPassContext options)
    {
        var caller = (Call)result["caller"];
        var callee = (Call)result["callee"];
        var caller_fusion = (Fusion)result["caller_fusion"];
        var callee_fusion = (Fusion)result["callee_fusion"];
        var callee_inputs = (IReadOnlyList<Expr>)result["callee_inputs"];

        if (usedByReslut[callee].Count() > 1)
        {
            return null;
        }

        var candidate_fusions = new HashSet<Fusion>() { caller_fusion, callee_fusion };

        if (!candidateFusionCheckCallBack(candidate_fusions))
        {
            return null;
        }

        // 1. merge new fusion
        var merged_fusion = ProcessMergeSingleInputFusion(mergedFusionRewriteCallBack, caller, callee, caller_fusion, callee_fusion);

        if (mergedFusionCheckCallBack(merged_fusion, candidate_fusions))
        {
            var new_call = new Call(merged_fusion, callee_inputs.ToArray());
            return new_call;
        }

        candidateFusionRecordCallBack(candidate_fusions);
        return null;
    }

    private Fusion ProcessMergeSingleInputFusion(Func<Expr, Expr> mergedFusionRewriteCallBack, Call caller, Call callee, Fusion caller_fusion, Fusion callee_fusion)
    {
        // 1. replace the caller_fusion input_var with the callee_fusion body
        var merger = new FusionMerger(caller_fusion.Parameters[0], callee_fusion.Body);
        var merged_fusion_body = merger.Clone(caller_fusion.Body, default);

        // 2. run call back.
        merged_fusion_body = mergedFusionRewriteCallBack(merged_fusion_body);
        if (!CompilerServices.InferenceType(merged_fusion_body))
        {
            throw new InvalidOperationException("Merged Fusion Type Infer Error!");
        }

        var fusionParams = callee_fusion.Parameters.AsValueEnumerable().Select(x => (Var)merger.ExprMemo[x]).ToArray();
        return new Fusion($"{caller_fusion.Name}_{callee_fusion.Name}", ModuleKind, merged_fusion_body, fusionParams);
    }

    private sealed class FusionMerger : ExprCloner<Unit>
    {
        private readonly Var _callerParam;
        private readonly Expr _calleeBody;

        public FusionMerger(Var callerParam, Expr calleeBody)
        {
            _callerParam = callerParam;
            _calleeBody = calleeBody;
        }

        protected override Expr VisitLeafVar(Var expr, Unit context)
        {
            if (ReferenceEquals(expr, _callerParam))
            {
                return Visit(_calleeBody, context);
            }

            return base.VisitLeafVar(expr, context);
        }
    }
}

/// <summary>
/// <see cref="ShortCutFusionMergeRuleLeft"/>.
/// </summary>
public class ShortCutFusionMergeRuleRight : ShortCutFusionMergeRuleLeft
{
    /// <inheritdoc/>
    public override Pattern CreatePattern(string targetModuleKind) => CreatePattern(targetModuleKind, false);
}

/// <summary>
///              x                       x
///         |         \                  |
///  v1 = fusion1(x)   |         => fusion2_1(x)
///          \          |
///            \       /
///         v2 = fusion2(v1,x).
/// ---------------------------
///         x                             x        y
///         |                             |       /
///  v1 = fusion1(x)     y       =>  fusion2_1(x,y)
///          \          |
///            \       /
///         v2 = fusion2(v1,y).
/// </summary>
public class ShortCutFusionMergeRuleLeft : IMergeRewriteRule
{
    private Pattern? _pattern;

    /// <summary>
    /// Gets the matched fusion module kind.
    /// </summary>
    public virtual string ModuleKind => Callable.StackVMModuleKind;

    /// <inheritdoc/>
    public IPattern Pattern => _pattern ??= CreatePattern(ModuleKind);

    /// <inheritdoc/>
    public virtual Pattern CreatePattern(string targetModuleKind) => CreatePattern(targetModuleKind, true);

    /// <inheritdoc/>
    public Expr? GetReplace(
      Func<Expr, Expr> mergedFusionRewriteCallBack,
      Func<Fusion, HashSet<Fusion>, bool> mergedFusionCheckCallBack,
      Func<HashSet<Fusion>, bool> candidateFusionCheckCallBack,
      Action<HashSet<Fusion>> candidateFusionRecordCallBack,
      IExprUserAnalysisResult usedByReslut,
      IMatchResult result,
      RunPassContext options)
    {
        var caller = (Call)result["caller"];
        var callee = (Call)result["callee"];
        var callerFusion = (Fusion)result["callerFusion"];
        var calleeFusion = (Fusion)result["calleeFusion"];
        var callerInputs = (IReadOnlyList<Expr>)result["callerInputs"];
        bool calleeInLeft = false;
        if (object.ReferenceEquals(callerInputs[0], callee))
        {
            calleeInLeft = true;
        }

        // can't merge left right have same input: caller(callee(x),callee(x))
        if (object.ReferenceEquals(callerInputs[calleeInLeft ? 1 : 0], callee))
        {
            return null;
        }

        var calleeInput = (Expr)result["calleeInput"];
        var callerOtherInput = (Expr)result["callerOtherInput"];

        var calleeInputUsers = new HashSet<Expr>(usedByReslut[calleeInput], ReferenceEqualityComparer.Instance);
        var calleeUsers = new HashSet<Expr>(usedByReslut[callee], ReferenceEqualityComparer.Instance);
        if (object.ReferenceEquals(calleeInput, callerOtherInput))
        {
            // case : caller(callee(x),x)
            // 1. callee input only usedby callee and caller
            if (calleeInputUsers.Count != 2)
            {
                return null;
            }

            if (!calleeInputUsers.Remove(callee) || !calleeInputUsers.Remove(caller))
            {
                return null;
            }
        }
        else
        {
            // case : caller(callee(x),y)
            if (!calleeUsers.Contains(caller))
            {
                return null;
            }
        }

        // 2. callee only usedby caller
        if (calleeUsers.Count != 1)
        {
            return null;
        }

        var candidateFusions = new HashSet<Fusion>() { callerFusion, calleeFusion };

        if (!candidateFusionCheckCallBack(candidateFusions))
        {
            return null;
        }

        // 1. merge new fusion
        var (mergedFusion, callParams) = ProcessMergeFusion(mergedFusionRewriteCallBack, calleeInput, callerOtherInput, caller, callee, callerFusion, calleeFusion, calleeInLeft);

        if (mergedFusionCheckCallBack(mergedFusion, candidateFusions))
        {
            var newCall = new Call(mergedFusion, callParams.ToArray());
            return newCall;
        }

        candidateFusionRecordCallBack(candidateFusions);
        return null;
    }

    /// <summary>
    /// create Pattern with position.
    /// </summary>
    /// <param name="targetModuleKind">module kind.</param>
    /// <param name="left">position.</param>
    protected Pattern CreatePattern(string targetModuleKind, bool left)
    {
        var calleeInput = IsWildcard("calleeInput");
        var callerOtherInput = IsWildcard("callerOtherInput");
        var calleePattern = IsCall(
          "callee",
          IsFusion(
              "calleeFusion",
              targetModuleKind,
              IsWildcard(),
              IsVArgs(IsWildcard())),
          calleeInput);
        var callerPatternLeft = IsCall(
          "caller",
          IsFusion(
              "callerFusion",
              targetModuleKind,
              IsWildcard(),
              IsVArgs(IsWildcard(), IsWildcard())),
          IsVArgs("callerInputs", new Pattern[] { calleePattern, callerOtherInput }));
        var callerPatternRight = IsCall(
          "caller",
          IsFusion(
              "callerFusion",
              targetModuleKind,
              IsWildcard(),
              IsVArgs(IsWildcard(), IsWildcard())),
          IsVArgs("callerInputs", new Pattern[] { callerOtherInput, calleePattern }));
        return left ? IsAlt(callerPatternLeft, callerPatternRight) : IsAlt(callerPatternRight, callerPatternLeft);
    }

    private (Fusion Fusion, List<Expr> Inputs) ProcessMergeFusion(Func<Expr, Expr> mergedFusionRewriteCallBack, Expr calleeInput, Expr callerOtherInput, Call caller, Call callee, Fusion callerFusion, Fusion calleeFusion, bool calleeInLeft)
    {
        bool calleeInputIsCallerOtherInput = ReferenceEquals(calleeInput, callerOtherInput);
        var calleeParam = calleeFusion.Parameters[0];
        var callerParam = calleeInLeft ? callerFusion.Parameters[0] : callerFusion.Parameters[1];
        var callerOtherParam = calleeInLeft ? callerFusion.Parameters[1] : callerFusion.Parameters[0];

        // 1. replace the caller_fusion input_var with the callee_fusion body
        var merger = new FusionMerger(calleeInputIsCallerOtherInput, callerParam, callerOtherParam, calleeParam, calleeFusion.Body);
        var merged_fusion_body = merger.Clone(callerFusion.Body, default);

        // 2. run call back.
        merged_fusion_body = mergedFusionRewriteCallBack(merged_fusion_body);
        if (!CompilerServices.InferenceType(merged_fusion_body))
        {
            throw new InvalidOperationException("Merged Fusion Type Infer Error!");
        }

        var callParams = new List<Expr>() { calleeInput };
        var fusionParams = new List<Var>() { (Var)merger.ExprMemo[calleeParam] };
        if (!calleeInputIsCallerOtherInput)
        {
            var callerOtherParamClone = (Var)merger.ExprMemo[callerOtherParam];
            if (calleeInLeft)
            {
                fusionParams.Add(callerOtherParamClone);
                callParams.Add(callerOtherInput);
            }
            else
            {
                fusionParams.Insert(0, callerOtherParamClone);
                callParams.Insert(0, callerOtherInput);
            }
        }

        return (new Fusion(
            $"{callerFusion.Name}_{calleeFusion.Name}",
            ModuleKind,
            merged_fusion_body,
            fusionParams.ToArray()),
            callParams);
    }

    private sealed class FusionMerger : ExprCloner<Unit>
    {
        private readonly bool _calleeInputIsCallerOtherInput;
        private readonly Var _callerParam;
        private readonly Var _callerOtherParam;
        private readonly Var _calleeParam;
        private readonly Expr _calleeBody;

        public FusionMerger(bool calleeInputIsCallerOtherInput, Var callerParam, Var callerOtherParam, Var calleeParam, Expr calleeBody)
        {
            _calleeInputIsCallerOtherInput = calleeInputIsCallerOtherInput;
            _callerParam = callerParam;
            _callerOtherParam = callerOtherParam;
            _calleeParam = calleeParam;
            _calleeBody = calleeBody;
        }

        protected override Expr VisitLeafVar(Var expr, Unit context)
        {
            if (ReferenceEquals(expr, _callerParam))
            {
                return Visit(_calleeBody, context);
            }

            // when call(fusion(x),x) only preserve one var as parmeters
            if (_calleeInputIsCallerOtherInput
                && ReferenceEquals(expr, _callerOtherParam))
            {
                return Visit(_calleeParam, context);
            }

            return base.VisitLeafVar(expr, context);
        }
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
public class SameInputFusionMergeRule : IMergeRewriteRule
{
    private Pattern? _pattern;

    /// <summary>
    /// Gets get ModuleKind.
    /// </summary>
    public virtual string ModuleKind => Callable.StackVMModuleKind;

    /// <inheritdoc/>
    public IPattern Pattern => _pattern ??= CreatePattern(ModuleKind);

    /// <inheritdoc/>
    public virtual Pattern CreatePattern(string target_module_kind)
    {
        var inputPat = IsWildcard("input");

        var callerPattern = IsCall(
            "caller",
            IsFusion(
                "caller_fusion",
                target_module_kind,
                IsWildcard(),
                IsVArgsRepeat(() => IsWildcard())),
            IsVArgsRepeat("caller_inputs", (ReadOnlySpan<Expr> caller_inputs) =>
            {
                var callee_patterns = new List<Pattern>();
                for (int i = 0; i < caller_inputs.Length; i++)
                {
                    /*
                    note the OrPattern need care the order.
                    pat  = or(input,call(input))
                    expr = (call(x), x)
                    the input will first match to `call(x)`, then can't match the x.
                    */
                    callee_patterns.Add(
                      IsAlt(
                        IsCall(
                            $"callee_{i}",
                            IsFusion($"callee_fusion_{i}", target_module_kind, IsWildcard(), IsVArgs(IsWildcard())),
                            inputPat),
                        inputPat));
                }

                return callee_patterns;
            }));
        return callerPattern;
    }

    /// <inheritdoc/>
    public Expr? GetReplace(
      Func<Expr, Expr> mergedFusionRewriteCallBack,
      Func<Fusion, HashSet<Fusion>, bool> mergedFusionCheckCallBack,
      Func<HashSet<Fusion>, bool> candidateFusionCheckCallBack,
      Action<HashSet<Fusion>> candidateFusionRecordCallBack,
      IExprUserAnalysisResult usedByReslut,
      IMatchResult result,
      RunPassContext options)
    {
        var caller = (Expr)result["caller"];
        var caller_fusion = (Fusion)result["caller_fusion"];
        var caller_inputs = (IReadOnlyList<Expr>)result["caller_inputs"];
        var input = (Expr)result["input"];

        // it will match  fusion2(fusion1(x,y)), then  input => fusion1(x,y), for merge cycle first so skip it.
        // if (input is Call { Target: Fusion input_fusion } && input_fusion.Arguments.Count != 1)
        //     return false;
        if (caller_inputs.Count == 1)
        {
            /* when match, skip.
                x
                |
              f1(x)
             */
            if (caller_inputs[0] is not Call { Target: Fusion candidate_fusion })
            {
                return false;
            }

            /* skip.
   \   |   /
 v0 = f0(x,y,z)
     |
 v1 = f1(v0)
*/
            if (candidate_fusion.Parameters.Length > 1)
            {
                return false;
            }

            /* skip when it have other user.
         v0 = f0(x)
       /          |
v2 =f2(v0)      v1 = f1(v0)
*/

            // todo. now can't find the mini case.
            if (usedByReslut[caller_inputs[0]].Count() > 1)
            {
                return false;
            }

            /* pass.
                x
                |
              f1(x)
                |
              f2(x)
            */
        }

        /*  1. when cycle like:
            x                x                  x                 x
           /  \           /  |  \         /  |  \      \          |
       f1(x)  f2(x)     | f1(x) f2(x)   |   |   f1(x) f2(x)     f1(x)
           \  /         \   |   /       \   |   /      /         / \
          f3(x,y)       f3(x,y,z)        f3(x,x,y,z)          f2(y,y)
        */
        if (caller_inputs.Count > 1)
        {
            var input_users = new HashSet<Expr>(usedByReslut[input], ReferenceEqualityComparer.Instance);

            // 1. remove the all mid fusion users.
            foreach (var caller_input in new HashSet<Expr>(caller_inputs, ReferenceEqualityComparer.Instance))
            {
                if (!object.ReferenceEquals(caller_input, input))
                {
                    if (!input_users.Remove(caller_input))
                    {
                        return false;
                    }
                }
            }

            // 2. if have more than one users, return false.
            if (input_users.Count > 1)
            {
                return false;
            }

            // 3. final user must be caller.
            if (input_users.Count == 1 && !input_users.Remove(caller))
            {
                return false;
            }

            // 4. check the caller input not usedby other call.
            /* eg. the f2(x) usedby f4.
                      x
                    /  \
                f1(x)  f2(x)
                    \  /     \
                    f3(x,y)  f4(x)
            */
            foreach (var caller_input in caller_inputs)
            {
                if (!object.ReferenceEquals(caller_input, input))
                {
                    var caller_input_users = new HashSet<Expr>(usedByReslut[caller_input], ReferenceEqualityComparer.Instance);
                    if (!caller_input_users.Remove(caller))
                    {
                        return false;
                    }

                    if (caller_input_users.Count > 0)
                    {
                        return false;
                    }
                }
            }
        }

        if (!ProcessFusionMerge(
            mergedFusionRewriteCallBack,
            candidateFusionCheckCallBack,
            caller,
            caller_fusion,
            caller_inputs,
            input,
            result,
            out var candidate_fusions,
            out var merged_fusion))
        {
            return null;
        }

        if (mergedFusionCheckCallBack(merged_fusion, candidate_fusions))
        {
            var new_call = new Call(merged_fusion, input);
            return new_call;
        }
        else
        {
            candidateFusionRecordCallBack(candidate_fusions);
        }

        return null;
    }

    private bool ProcessFusionMerge(Func<Expr, Expr> mergedFusionRewriteCallBack, Func<HashSet<Fusion>, bool> candidate_fusion_checker, Expr caller, Fusion caller_fusion, IReadOnlyList<Expr> caller_inputs, Expr input, IMatchResult result, out HashSet<Fusion> candidate_fusions, out Fusion merged_fusion)
    {
        merged_fusion = null!;
        candidate_fusions = new() { caller_fusion };
        var calleeBodyMap = new Dictionary<Var, Expr>(ReferenceEqualityComparer.Instance);
        var multiVarMap = new Dictionary<Var, Var>();
        if (input.CheckedType is null)
        {
            CompilerServices.InferenceType(input);
        }

        var new_fusion_input_var = new Var(input.CheckedType!);
        string new_fusion_name = $"{caller_fusion.Name}";

        bool has_fusion_for_merge = false; // because of it can match fusion(var)

        for (int i = 0; i < caller_inputs.Count; i++)
        {
            if (caller_inputs[i] is Call { Target: Fusion })
            {
                if (result[$"callee_fusion_{i}"] is not Fusion callee_fusion)
                {
                    return false;
                }

                calleeBodyMap.Add(caller_fusion.Parameters[i], callee_fusion.Body);
                has_fusion_for_merge = true;

                // avoid same callee fusion
                if (candidate_fusions.Add(callee_fusion))
                {
                    new_fusion_name += "_" + callee_fusion.Name;
                    multiVarMap.Add(callee_fusion.Parameters[0], new_fusion_input_var);
                }
            }
            else
            {
                multiVarMap.Add(caller_fusion.Parameters[i], new_fusion_input_var);
            }
        }

        if (!has_fusion_for_merge)
        {
            return false;
        }

        if (!candidate_fusion_checker(candidate_fusions))
        {
            return false;
        }

        // 1. replace the caller_fusion input_var with the callee_fusion_i body
        var merger = new IMergeRewriteRule.FusionMerger(calleeBodyMap, multiVarMap);
        var merged_fusion_body = merger.Clone(caller_fusion.Body, default);

        // 2. run call back.
        merged_fusion_body = mergedFusionRewriteCallBack(merged_fusion_body);
        if (!CompilerServices.InferenceType(merged_fusion_body))
        {
            throw new InvalidOperationException("Merged Fusion Type Infer Error!");
        }

        merged_fusion = new Fusion(new_fusion_name, ModuleKind, merged_fusion_body, new[] { new_fusion_input_var });

        return true;
    }

    private sealed class FusionMerger : ExprCloner<Unit>
    {
        private readonly IReadOnlyDictionary<Var, Expr> _calleeBodyMap;
        private readonly IReadOnlyDictionary<Var, Var> _multiVarMap;

        public FusionMerger(IReadOnlyDictionary<Var, Expr> calleeBodyMap, IReadOnlyDictionary<Var, Var> multiVarMap)
        {
            _calleeBodyMap = calleeBodyMap;
            _multiVarMap = multiVarMap;
        }

        protected override Expr VisitLeafVar(Var expr, Unit context)
        {
            if (_calleeBodyMap.TryGetValue(expr, out var callee))
            {
                return Visit(callee, context);
            }

            if (_multiVarMap.TryGetValue(expr, out var newVar))
            {
                return newVar;
            }

            throw new InvalidOperationException();
        }
    }
}
