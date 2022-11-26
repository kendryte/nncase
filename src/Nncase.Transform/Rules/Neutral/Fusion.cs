using System;
using System.Linq;
using Nncase.CodeGen;
using Nncase.IR;
using Nncase.PatternMatch;
using static Nncase.PatternMatch.Utility;
using static Nncase.Utilities.ReplaceUtility;
namespace Nncase.Transform.Rules.Neutral;

public abstract class FusionMaker : RewriteRule<Pattern>
{
    public virtual string Name { get; } = "SingleInputFusion";

    public virtual string ModuleKind { get; } = "StackVM";

    private int count = 0;

    public string FullName => $"{Name}_{count++}";
}

[RuleGenerator]
public partial class SingleInputFusion<T, BeginT, EndT> : FusionMaker
    where T : Op
    where BeginT : Op
    where EndT : Op
{
    /// <inheritdoc/>
    public override Pattern Pattern { get; } = IsWildcardCall<EndT>("st", null!,
        IsWildcardCall<T>(null!, null!, (
            IsWildcardCall<BeginT>(null!, null!, IsWildcard("input")))));

    // replace input with var
    private Call? GetReplace(Call st, Expr input, RunPassOptions passOptions)
    {
        var arg = new Var("input0", input.CheckedType!);
        var body = ReplaceTarget(st, input, arg, passOptions.MatchOptions);
        var fusion = new Call(new Fusion(FullName, ModuleKind, body, new[] { arg }), input);
        return fusion;
        // options.SuppressPattern(st, Pattern);
        // return fusion;
    }
}

[RuleGenerator]
public partial class DoubleInputFusion<T, BeginT, EndT> : FusionMaker
    where T : Op
    where BeginT : Op
    where EndT : Op
{
    /// <inheritdoc/>
    public override Pattern Pattern { get; } = IsWildcardCall<EndT>("st", null!,
        IsWildcardCall<T>(null!, null!,
            IsWildcardCall<BeginT>(null!, null!, IsWildcard("lhs")),
            IsWildcardCall<BeginT>(null!, null!, IsWildcard("rhs"))));

    // replace input with var
    private Call GetReplace(Call st, Expr lhs, Expr rhs, RunPassOptions passOptions)
    {
        var arg0 = new Var("input0", lhs.CheckedType!);
        var arg1 = new Var("input1", rhs.CheckedType!);
        var tmpBody = ReplaceTarget(st, lhs, arg0, passOptions.MatchOptions);
        var body = ReplaceTarget(tmpBody, rhs, arg1, passOptions.MatchOptions);
        var fusion = new Call(new Fusion(FullName, ModuleKind, body, new[] { arg0, arg1 }), lhs, rhs);
        return fusion;
        // options.SuppressPattern(st, Pattern);
        // return fusion;
    }
}

[RuleGenerator]
public partial class DataTransferFusion<LoadT, StoreT> : FusionMaker
    where LoadT : Op
    where StoreT : Op
{
    /// <inheritdoc/>
    public override Pattern Pattern { get; } = IsWildcardCall<StoreT>("st", null!,
        IsWildcardCall<LoadT>(null!, null!, IsWildcard("input")));

    // replace input with var
    private Call? GetReplace(Call st, Expr input, RunPassOptions passOptions)
    {
        if ((st.Attribute & CallAttr.Fusion) != 0)
        {
            return null;
        }

        var arg = new Var("input0", input.CheckedType!);
        var body = ReplaceTarget(st, input, arg, passOptions.MatchOptions);
        return new Call(new Fusion(FullName, ModuleKind, body, new[] { arg }), input);
    }
}

[RuleGenerator]
public partial class FuseTwoFusion : RewriteRule<Pattern>
{
    /// <summary>
    /// module kind
    /// </summary>
    public virtual string ModuleKind { get; } = "StackVM";
    
    private Pattern? _calleePattern = null;
    private Pattern? _Pattern = null;

    /// <inheritdoc/>
    public override Pattern Pattern => _Pattern ??=
      IsCall(
        "caller",
        IsFusion("callerFuse",
            ModuleKind,
            IsWildcard(),
            WildcardVArgsPattern),
        ParamsWithArg(CalleePattern)
        );

    /// <inheritdoc/>
    public Pattern CalleePattern => _calleePattern ??=
        IsCall(
        "callee",
        IsFusion("calleeFuse",
            ModuleKind,
            IsWildcard(),
            WildcardVArgsPattern),
        WildcardVArgsPattern);

    // caller(callee, args..)
    private Call GetReplace(Call caller, Call callee, Fusion calleeFuse, Fusion callerFuse, RunPassOptions passOptions)
    {
        // find callee pos index in caller
        // input0  input1
        //    \      /
        //     caller
        // param[1] == callee, then index == 1
        var index = caller.Parameters.ToList().FindIndex(x =>
        {
            passOptions.MatchOptions.TryUpdateWithRewrite(ref x);
            return x == callee;
        });
        // get param var name from args index for rename
        var beReplacedVar = callerFuse.Parameters[index];

        // replace calleeFuse first param with callerParam which be removed
        var (calleeFuseBody, calleeFirstVar) = RenameFirstVar(calleeFuse, beReplacedVar.Name, passOptions.MatchOptions);
        var (newCalleeFuseBody, newCalleeParams) = RenameRestVar(calleeFuse.Parameters, calleeFuseBody, caller.Parameters.Count, passOptions.MatchOptions);

        // merge two body
        //     input1 input2 input3
        //        \     |      /
        // input0    callee  
        //   \        /         
        //     caller               
        var newBodyWithRedundancy = ReplaceTarget(callerFuse.Body, beReplacedVar, newCalleeFuseBody, passOptions.MatchOptions);
        // eliminate store load
        // fusion: load -> op1 -> op2 -> ... -> store
        var newBody = EliminateRedundancy(newBodyWithRedundancy, passOptions);

        var newParams = Merge(callerFuse.Parameters.ToArray(), index, calleeFirstVar, newCalleeParams);
        var newInputs = Merge(caller.Parameters, index, callee.Parameters[0], callee.Parameters.ToArray()[1..]);
        return new Call(new Fusion("FuseTwoFusion", "k510", newBody, newParams), newInputs);
    }

    /// <summary>
    /// e.g. load -> conv -> [store -> load] -> act -> store =>
    ///      load -> conv -> act -> store
    /// </summary>
    /// <param name="newBodyWithRedundancy"></param>
    /// <param name="passOptions"></param>
    /// <returns></returns>
    /// <exception cref="InvalidOperationException"></exception>
    virtual public Expr EliminateRedundancy(Expr newBodyWithRedundancy, RunPassOptions passOptions)
    {
        throw new InvalidOperationException("EliminateRedundancy Not Impl");
    }

    public T[] Merge<T>(IRArray<T> callerExprList, int index, T firstInCallee,
        IRArray<Expr> newCalleeExprList) where T : Expr =>
        Merge<T>(callerExprList.ToArray(), index, firstInCallee, newCalleeExprList.ToArray());

    public T[] Merge<T>(T[] callerExprList, int index, T expr, Expr[] newCalleeExprList)
        where T : Expr
    {
        var newCallerExprList = ReplacePos(callerExprList, expr, index);
        return newCallerExprList.Concat(newCalleeExprList).Select(x => (T)x).ToArray();
    }

    /// <summary>
    /// rename first var
    /// input0 input1  input0 input1 input2
    ///   \     /         \      |      /
    ///   caller               callee
    /// when callee == input0, then not replace and return
    /// else replace input1 with callee first var
    /// else result:
    /// input0 input1  input1 input1 input2
    ///   \     /         \      |      /
    ///   caller               callee
    /// right input will be rename in RenameRestVar 
    /// </summary>
    /// <param name="calleeFuse"></param>
    /// <param name="beReplacedVarName"></param>
    /// <returns></returns>
    public (Expr, Var) RenameFirstVar(Fusion calleeFuse, string beReplacedVarName, MatchOptions matchOptions)
    {
        // if callee first var name is not same as beReplaceVarName, then replace old var with newVar in calleeFuseBody
        var newVar = new Var(beReplacedVarName, calleeFuse.Parameters[0].CheckedType);
        return calleeFuse.Parameters[0].Name == beReplacedVarName
            ? (calleeFuse.Body, calleeFuse.Parameters[0])
            : (ReplaceTarget(
                calleeFuse.Body,
                calleeFuse.Parameters[0],
                // create a new var
                newVar, matchOptions), newVar);
    }

    /// <summary>
    /// rename rest var
    /// if count of calleeParams > 1, then rename
    /// input0 input1  input1 input2 input3
    ///   \     /         \      |     |
    ///   caller              callee
    /// right input will be rename in RenameRestVar 
    /// </summary>
    /// <param name="calleeFuseParams"></param>
    /// <param name="calleeFuseBody"></param>
    /// <param name="callerParamCount"></param>
    /// <param name="matchOptions"></param>
    /// <returns></returns>
    public (Expr, Var[]) RenameRestVar(IRArray<Var> calleeFuseParams, Expr calleeFuseBody, int callerParamCount, MatchOptions matchOptions)
    {
        var newVarRange = Enumerable.Range(1, calleeFuseParams.Count - 1);
        var newInputCountBase = callerParamCount;
        // replace other param
        var newCalleeParams = newVarRange
            .Select(i => calleeFuseParams[i] with { Name = "input" + (newInputCountBase - 1 + i) })
            .ToArray();
        // rename var in callee
        var newCalleeFuseBody = newVarRange
            .Aggregate(calleeFuseBody, (expr, i) =>
                ReplaceTarget(
                    expr,
                    calleeFuseParams[i],
                    newCalleeParams[i - 1],
                    matchOptions));
        return (newCalleeFuseBody, newCalleeParams);
    }
}