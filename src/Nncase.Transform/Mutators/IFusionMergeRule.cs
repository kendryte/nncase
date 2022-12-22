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
public interface IMergeRewriteRule
{
    /// <summary>
    /// Gets pattern.
    /// </summary>
    IPattern Pattern { get; }

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
    Expr? GetReplace(
      Func<Expr, Expr> mergedFusionRewriteCallBack,
      Func<Fusion, HashSet<Fusion>, bool> mergedFusionCheckCallBack,
      Func<HashSet<Fusion>, bool> candidateFusionCheckCallBack,
      Action<HashSet<Fusion>> candidateFusionRecordCallBack,
      IUsedByResult usedByReslut, IMatchResult result, RunPassOptions options);
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
    private readonly Pattern? _pattern;

    /// <summary>
    /// Gets the matched fusion module kind.
    /// </summary>
    public virtual string ModuleKind => Callable.StackVMModuleKind;

    /// <inheritdoc/>
    public IPattern Pattern => _pattern ?? CreatePattern(ModuleKind);

    /// <inheritdoc/>
    public virtual Pattern CreatePattern(string target_module_kind)
    {
        var CalleePattern = IsCall(
          "callee",
          IsFusion(
              "callee_fusion",
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
            IsFusion(
                "caller_fusion",
              target_module_kind,
              IsWildcard(),
              IsVArgs(IsWildcard())),
            CalleePattern);
        return CallerPattern;
    }

    /// <inheritdoc/>
    public Expr? GetReplace(
      Func<Expr, Expr> mergedFusionRewriteCallBack,
      Func<Fusion, HashSet<Fusion>, bool> mergedFusionCheckCallBack,
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
            var new_call = new Call(merged_fusion, ImmutableArray.CreateRange(callee_inputs));

            // 1. transfer the caller usedby info to new_call
            usedByReslut.Transfer(caller, new_call);

            // 2. clear all caller's and callee's usedy info
            usedByReslut.Clear(caller_fusion, caller);
            usedByReslut.Clear(callee, caller);
            usedByReslut.Clear(callee_fusion, callee);
            foreach (var callee_input in callee_inputs)
            {
                usedByReslut.Clear(callee_input, callee);
            }

            // 3. reset the input usedby
            foreach (var callee_input in callee_inputs)
            {
                usedByReslut.Add(callee_input, new_call);
            }

            usedByReslut.Add(merged_fusion, new_call);
            return new_call;
        }

        candidateFusionRecordCallBack(candidate_fusions);
        return null;
    }

    private Fusion ProcessMergeSingleInputFusion(Func<Expr, Expr> mergedFusionRewriteCallBack, Call caller, Call callee, Fusion caller_fusion, Fusion callee_fusion)
    {
        // 1. replace the caller_fusion input_var with the callee_fusion body
        var merged_fusion_body = Transform.Mutator.Substitute(
          e => object.ReferenceEquals(e, caller_fusion.Parameters[0]) ?
           callee_fusion.Body :
           null)().Visit(caller_fusion.Body);

        // 2. run call back.
        merged_fusion_body = mergedFusionRewriteCallBack(merged_fusion_body);
        if (!CompilerServices.InferenceType(merged_fusion_body))
        {
            throw new InvalidOperationException("Merged Fusion Type Infer Error!");
        }

        return new Fusion($"{caller_fusion.Name}_{callee_fusion.Name}", ModuleKind, merged_fusion_body, callee_fusion.Parameters);
    }
}

/// <summary>
///              x
///         |         \
///  v1 = fusion1(x)   |
///          \          |
///            \       /
///         v2 = fusion2(v1,x).
/// </summary>
public class ShortCutFusionMergeRule : IMergeRewriteRule
{
    private readonly Pattern? _pattern;

    /// <summary>
    /// Gets the matched fusion module kind.
    /// </summary>
    public virtual string ModuleKind => Callable.StackVMModuleKind;

    /// <inheritdoc/>
    public IPattern Pattern => _pattern ?? CreatePattern(ModuleKind);

    /// <inheritdoc/>
    public virtual Pattern CreatePattern(string target_module_kind)
    {
        var input = IsWildcard("callee_input");
        var CalleePattern = IsCall(
          "callee",
          IsFusion(
              "callee_fusion",
            target_module_kind,
            IsWildcard(),
            IsVArgs(IsWildcard())),
          input);
        var CallerPattern_left = IsCall(
          "caller",
          IsFusion(
              "caller_fusion",
            target_module_kind,
            IsWildcard(),
            IsVArgs(IsWildcard(), IsWildcard())),
          IsVArgs("caller_inputs", new Pattern[] { CalleePattern, input }));
        var CallerPattern_right = IsCall(
          "caller",
          IsFusion(
              "caller_fusion",
            target_module_kind,
            IsWildcard(),
            IsVArgs(IsWildcard(), IsWildcard())),
          IsVArgs("caller_inputs", new Pattern[] { input, CalleePattern }));
        return IsAlt(CallerPattern_left, CallerPattern_right);
    }

    /// <inheritdoc/>
    public Expr? GetReplace(
      Func<Expr, Expr> mergedFusionRewriteCallBack,
      Func<Fusion, HashSet<Fusion>, bool> mergedFusionCheckCallBack,
      Func<HashSet<Fusion>, bool> candidateFusionCheckCallBack,
      Action<HashSet<Fusion>> candidateFusionRecordCallBack,
      IUsedByResult usedByReslut, IMatchResult result, RunPassOptions options)
    {
        var caller = (Call)result["caller"];
        var callee = (Call)result["callee"];
        var caller_fusion = (Fusion)result["caller_fusion"];
        var callee_fusion = (Fusion)result["callee_fusion"];
        var caller_inputs = (IReadOnlyList<Expr>)result["caller_inputs"];
        bool callee_in_left = false;
        if (object.ReferenceEquals(caller_inputs[0], callee))
        {
            callee_in_left = true;
        }

        var callee_input = (Expr)result["callee_input"];

        // 1. callee input only usedby callee and caller
        var callee_input_users = new HashSet<Expr>(usedByReslut.Get(callee_input), ReferenceEqualityComparer.Instance);
        if (callee_input_users.Count != 2)
        {
            return null;
        }

        if (!callee_input_users.Remove(callee) || !callee_input_users.Remove(caller))
        {
            return null;
        }

        // 2. callee only usedby caller
        var callee_users = new HashSet<Expr>(usedByReslut.Get(callee), ReferenceEqualityComparer.Instance);
        if (callee_users.Count != 1)
        {
            return null;
        }

        var candidate_fusions = new HashSet<Fusion>() { caller_fusion, callee_fusion };

        if (!candidateFusionCheckCallBack(candidate_fusions))
        {
            return null;
        }

        // 1. merge new fusion
        var merged_fusion = ProcessMergeFusion(mergedFusionRewriteCallBack, caller, callee, caller_fusion, callee_fusion, callee_in_left);

        if (mergedFusionCheckCallBack(merged_fusion, candidate_fusions))
        {
            var new_call = new Call(merged_fusion, ImmutableArray.Create(callee_input));

            // 1. transfer the caller usedby info to new_call
            usedByReslut.Transfer(caller, new_call);

            // 2. clear all caller's and callee's usedy info
            usedByReslut.Clear(caller_fusion, caller);
            usedByReslut.Clear(callee, caller);
            usedByReslut.Clear(callee_fusion, callee);
            usedByReslut.Clear(callee_input, callee);

            // 3. reset the input usedby
            usedByReslut.Add(callee_input, new_call);
            usedByReslut.Add(merged_fusion, new_call);
            return new_call;
        }

        candidateFusionRecordCallBack(candidate_fusions);
        return null;
    }

    private Fusion ProcessMergeFusion(Func<Expr, Expr> mergedFusionRewriteCallBack, Call caller, Call callee, Fusion caller_fusion, Fusion callee_fusion, bool callee_in_left)
    {
        // 1. replace the caller_fusion input_var with the callee_fusion body
        var merged_fusion_body = Transform.Mutator.Substitute(
          e =>
          {
              if (object.ReferenceEquals(e, callee_in_left ? caller_fusion.Parameters[0] : caller_fusion.Parameters[1]))
              {
                  return callee_fusion.Body;
              }

              return null;
          })().Visit(caller_fusion.Body);

        merged_fusion_body = Transform.Mutator.Substitute(
          e =>
          {
              if (object.ReferenceEquals(e, callee_in_left ? caller_fusion.Parameters[1] : caller_fusion.Parameters[0]))
              {
                  return callee_fusion.Parameters[0];
              }

              return null;
          })().Visit(merged_fusion_body);

        // 2. run call back.
        merged_fusion_body = mergedFusionRewriteCallBack(merged_fusion_body);
        if (!CompilerServices.InferenceType(merged_fusion_body))
        {
            throw new InvalidOperationException("Merged Fusion Type Infer Error!");
        }

        return new Fusion($"{caller_fusion.Name}_{callee_fusion.Name}", ModuleKind, merged_fusion_body, callee_fusion.Parameters);
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
    private readonly Pattern? _pattern;

    /// <summary>
    /// Gets get ModuleKind.
    /// </summary>
    public virtual string ModuleKind => Callable.StackVMModuleKind;

    /// <inheritdoc/>
    public IPattern Pattern => _pattern ?? CreatePattern(ModuleKind);

    /// <inheritdoc/>
    public virtual Pattern CreatePattern(string target_module_kind)
    {
        var inputPat = IsWildcard("input");

        var CallerPattern = IsCall(
            "caller",
            IsFusion(
                "caller_fusion",
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
                        IsCall(
                            $"callee_{i}",
                          IsFusion($"callee_fusion_{i}", target_module_kind, IsWildcard(), IsVArgs(IsWildcard())),
                          inputPat),
                        inputPat));
                }

                return new(callee_patterns);
            }));
        return CallerPattern;
    }

    /// <inheritdoc/>
    public Expr? GetReplace(
      Func<Expr, Expr> mergedFusionRewriteCallBack,
      Func<Fusion, HashSet<Fusion>, bool> mergedFusionCheckCallBack,
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
            {
                return false;
            }

            /* skip.
   \   |   /
 v0 = f0(x,y,z)
     |
 v1 = f1(v0)
*/
            if (candidate_fusion.Parameters.Count > 1)
            {
                return false;
            }

            /* skip when it have other user.
         v0 = f0(x)
       /          |
v2 =f2(v0)      v1 = f1(v0)
*/

            // todo. now can't find the mini case.
            if (usedByReslut.Get(caller_inputs[0]).Count > 1)
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
                    var caller_input_users = new HashSet<Expr>(usedByReslut.Get(caller_input), ReferenceEqualityComparer.Instance);
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

        if (!ProcessFusionMerge(mergedFusionRewriteCallBack, candidateFusionCheckCallBack, caller, caller_fusion, caller_inputs,
           input, result,
           out var candidate_fusions, out var merged_fusion))
        {
            return null;
        }

        if (mergedFusionCheckCallBack(merged_fusion, candidate_fusions))
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
                {
                    usedByReslut.Clear(callee_fusion, caller_inputs[i]);
                }

                if (!object.ReferenceEquals(caller_inputs[i], input))
                {
                    usedByReslut.Clear(input, caller_inputs[i]);
                }
            }

            // 3. reset the input usedby
            usedByReslut.Add(input, new_call);
            usedByReslut.Add(merged_fusion, new_call);
            return new_call;
        }
        else
        {
            candidateFusionRecordCallBack(candidate_fusions);
        }

        return null;
    }

    private bool ProcessFusionMerge(Func<Expr, Expr> mergedFusionRewriteCallBack, Func<HashSet<Fusion>, bool> candidate_fusion_checker, Expr caller, Fusion caller_fusion,
      IReadOnlyList<Expr> caller_inputs, Expr input, IMatchResult result, out HashSet<Fusion> candidate_fusions, out Fusion merged_fusion)
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
                new_fusion_name += "_" + callee_fusion.Name;
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
        {
            return false;
        }

        if (!candidate_fusion_checker(candidate_fusions))
        {
            return false;
        }

        // 1. replace the caller_fusion input_var with the callee_fusion_i body
        Expr merged_fusion_body = caller_fusion.Body;
        if (calleeBodyMap.Count > 0)
        {
            merged_fusion_body = Transform.Mutator.Substitute(e =>
            {
                if (e is Var ve && calleeBodyMap.TryGetValue(ve, out var new_e))
                {
                    return new_e;
                }

                return null;
            })().Visit(merged_fusion_body);
        }

        // 2. replace the all input var to new input var
        merged_fusion_body = Transform.Mutator.Substitute(e =>
        {
            if (e is Var ve && multiVarMap.TryGetValue(ve, out var new_e))
            {
                return new_e;
            }

            return null;
        })().Visit(merged_fusion_body);

        // 2. run call back.
        merged_fusion_body = mergedFusionRewriteCallBack(merged_fusion_body);
        if (!CompilerServices.InferenceType(merged_fusion_body))
        {
            throw new InvalidOperationException("Merged Fusion Type Infer Error!");
        }

        merged_fusion = new Fusion(new_fusion_name, ModuleKind, merged_fusion_body, new[] { new_fusion_input_var });

        return true;
    }
}
