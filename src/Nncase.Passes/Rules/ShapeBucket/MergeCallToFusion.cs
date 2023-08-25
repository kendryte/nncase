// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Xml;
using NetFabric.Hyperlinq;
using Nncase.IR;
using Nncase.IR.Tensors;
using Nncase.PatternMatch;
using static Nncase.Passes.Rules.ShapeBucket.ShapeBucketHelper;
using static Nncase.PatternMatch.F.Math;
using static Nncase.PatternMatch.Utility;
using static Nncase.Utilities.ReplaceUtility;

namespace Nncase.Passes.Rules.ShapeBucket;
public abstract class MergeFusionBase : RewriteRule<Pattern>
{
    protected int Counter { get; set; }

    public static bool AllConst(Call originCall)
    {
        // 暂时只能合并其他输入为const的
        if (originCall.Arguments.Length == 1)
        {
            return true;
        }

        var otherArgs = originCall.Arguments[1..].ToArray();
        if (otherArgs.All(x => x is Const || x is Marker { Target: Const }))
        {
            return true;
        }

        return false;
    }

    public bool ValidTarget(Expr target)
    {
        return CallValidator.ValidTarget(target);
    }
}

[RuleGenerator]
public partial class MergeNextMarkerToFusion : MergeFusionBase
{
    // 用于将fusion call 外部的marker合并进来
    public override Pattern Pattern => IsRangeOfMarker("marker", new MergeNextCallToFusion().FusionCall, IsWildcard());

    // 外部保留marker给下一个使用
    public Expr? GetReplace(Marker marker, Call fusionOuterCall, BucketFusion fusion, RunPassContext context)
    {
        if (fusion.Body is Marker)
        {
            return null;
        }

        // marker
        if (fusionOuterCall.Users.Count > 1 || marker.Users.Count > 1)
        {
            return null;
        }

        var result =
            marker.With(target: fusionOuterCall.With(target: fusion.With(body: marker.With(target: fusion.Body))));
        return result;
    }
}

[RuleGenerator]
public partial class MergePrevMarkerToFusion : MergeFusionBase
{
    public override Pattern Pattern => IsCall(
        "fusionOuterCall",
        IsFusion(
            "fusion",
            "stackvm",
            IsWildcard(),
            GenerateParameters(null)),
        GenerateParameters(null, IsRangeOfMarker("marker", IsWildcard(), IsWildcard())));

    // fusion(marker(xx)) { var } -> fusion(marker(xx)) { marker(var) }
    public Expr? GetReplace(Marker marker, Call fusionOuterCall, BucketFusion fusion)
    {
        var hasVisited = fusion.Parameters[0].Users.Where(u => u is not Fusion).All(u => u is Marker);
        if (hasVisited)
        {
            return null;
        }

        // 不更改原始的input中的marker，要拷贝到fusion里面，将所有的var替换为marker(var)
        // 同时将fusion的body中用到原始var的地方替换为marker(var)
        // MergeCall的时候是支持marker的
        var newBody = ReplaceExpr(fusion.Body, fusion.Parameters[0], marker.With(target: fusion.Parameters[0]));

        // 重新构建fusion
        var newFusion = fusion.With(body: newBody);

        // 返回新的call
        DumpIR(newFusion, $"{Counter++}_{fusion.Name}");
        return fusionOuterCall.With(target: newFusion);
    }
}

[RuleGenerator]
public partial class MergeNextCallToFusion : MergeFusionBase
{
    public Pattern FusionCall => IsCall(
        "fusionOuterCall",
        IsFusion(
            "fusion",
            "stackvm",
            IsWildcard(),
            GenerateParameters(null)),
        GenerateParameters(null));

    public override Pattern Pattern => IsCallWildcard(
        "nextCall",
        IsWildcard("target"),
        IsAlt(
            "maybeFusionCallMarker",
            FusionCall,
            IsRangeOfMarker(FusionCall, IsWildcard())));

    // nextCall(fusion(x)) -> fusion(nextCall(x))
    // nextCall(marker(fusion(x))) -> fusion(nextCall(marker(x)))
    public Expr? GetReplace(Call nextCall, Expr maybeFusionCallMarker, Expr target, Call fusionOuterCall, BucketFusion fusion)
    {
        var singleVar = CompileSession.CompileOptions.ShapeBucketOptions.VarMap.Values.SelectMany(x => x).OfType<Var>().ToHashSet().Count <= 1;
        if (!singleVar && nextCall.Arguments.ToArray().OfType<Call>().Count() > 1)
        {
            return null;
        }

        if (!ValidTarget(target))
        {
            return null;
        }

        // todo: only for single input, effect var must be same
        if (MultiUser(maybeFusionCallMarker))
        {
            return null;
        }

        // ref test TestMergeNextWithUserHasMultiUser
        if (MultiUser(nextCall))
        {
            // 会复制
            return null;
        }

        if (!AllConst(nextCall))
        {
            return null;
        }

        DumpIR(nextCall, $"{Counter}_{fusion.Name}_{target.GetType().Name}_origin");

        // 将call里面call fusion的部分替换为fusion的body
        var oldBody = fusion.Body;

        // 这里必须新构建一个Expr，不能使用原始的nextCall Replace掉参数，不然如果外面有marker,那么replace以后的call还是会被外面的marker引用，因此会出现重复的情况
        // arg0可能是marker，如果是marker的话不能替换marker的参数，而是重新构造marker
        Expr newBody = ReplaceCallParams(nextCall.Target, nextCall.Arguments.ToArray(), (0, (Expr)oldBody));

        // todo: 针对marker的测试
        if (nextCall.Users.Count == 1 && nextCall.Users.First() is Marker m)
        {
            newBody = m.With(target: newBody);
        }

        // 除了第一个参数的部分，其他参数可能会用到外面的东西，是不是可以作为var直接传进来??但是这会影响后面ToFusion的部分...

        // 更新fusion的body
        var newFusion = fusion.With(body: newBody);

        // 创建新的call，target为fusion，参数为fusion的参数 // todo:针对非const的情况要处理这里
        // 但是第一个参数要注意，如果有marker那么需要处理marker // 这里如果arg是marker的话则需要copy一份，不然会导致marker的user重复，进而复制了if
        // var newArgs = fusionOuterCall.Arguments.ToArray().Select(arg => arg is Marker m ? m.With() : arg).ToArray();
        var newArgs = fusionOuterCall.Arguments.ToArray().ToArray();
        var call = (Expr)nextCall.With(target: newFusion, arguments: newArgs);

        // 附加next call的外面marker
        DumpIR(call, $"{Counter++}_{fusion.Name}_{target.GetType().Name}_after");
        if (newBody.Users.Count > 1)
        {
            throw new InvalidOperationException($"{newFusion.Name} is Invalid");
        }

        ArgsChecker(newArgs);

        return call;
    }

    private static bool MultiUser(Expr call)
    {
        // 不是marker直接判断count
        if (call is Call && call.Users.Count > 1)
        {
            return true;
        }

        // 是marker那么判断marker的users
        if (call is Marker { Users.Count: > 1 })
        {
            return true;
        }

        // 不是marker那就没问题，一定不是多个user
        return false;
    }

    private bool SameEffectVar(Call originCall, Fusion fusion)
    {
        var array = MakeEffectVarArray(
            CompileSession,
            CompileSession.CompileOptions.ShapeBucketOptions.VarMap,
            originCall.Arguments[^1..].ToArray());
        if (fusion is BucketFusion varFusion)
        {
            if (array.Length != 0 && !Enumerable.SequenceEqual(varFusion.EffectVar, array))
            {
                return true;
            }
        }
        else
        {
            return true;
        }

        return false;
    }
}

[RuleGenerator]
public partial class MergePrevCallToFusion : MergeFusionBase
{
    private string _prevCallStr = string.Empty;

    public override Pattern Pattern => IsCall(
        "fusionOuterCall",
        IsFusion(
            "fusion",
            "stackvm",
            IsWildcard(),
            GenerateParameters(null)),
        GenerateParameters(
            null,
            IsWildcard()));

    public Pattern PrevCall(string prefix) => IsCallWildcard($"{prefix}PrevCall", IsWildcard($"{prefix}Target"));

    public Pattern MaybeMarker(string exprName, Pattern exprPatten) => IsAlt(
        exprName,
        IsRangeOfMarker(exprPatten, IsWildcard()),
        exprPatten);

    // 输入必须匹配marker，因为即便合并marker也是要在外面保留一份副本
    // fusion(marker(prevCall()) { var } -> fusion(var) { marker(prevCall()) }
    // fusion((prevCall()) { var } -> fusion(var) { prevCall() }

    // dfs
    // xx | marker(xx)不行, 会先匹配到xx
    // xx(marker) | xx 可以
    public Expr? GetReplace(Call fusionOuterCall, BucketFusion fusion)
    {
        // multi var的情况下，matmul的var一定是由输入构成，所以一定可以合并
        var (fusionArgsInfo, prevOutputMaybeMarker) = CollectInputsInfo(fusionOuterCall);
        if (fusionArgsInfo.Length == 0)
        {
            return null;
        }

        // FusionArgs
        var inputShouldBeMerge = CollectInputShouldBeMerge(fusionArgsInfo);
        var prefix = $"{Counter}_{_prevCallStr}_{fusion.Name}_origin";

        DumpIR(fusionOuterCall, prefix, printPrefix: "MergePrevCallToFusion");

        var indices = fusionArgsInfo.Select(x => x.Item2).ToHashSet();
        var fusionDict = fusionOuterCall.Arguments.ToArray().Zip(fusion.Parameters.ToArray())
            .Where((expr, i) => !indices.Contains(i))
            .ToDictionary(pair => pair.First, pair => pair.Second);

        // (InputArg -> NewFusionVar[]), InputArg is part of newArgs.
        var newVarsMap = MakeNewFusionVarsMap(fusionArgsInfo, fusionDict);

        // 所有要被合并的call替换args为Fusion的Var
        var newPrevCalls = MakeNewPrevCalls(inputShouldBeMerge, prevOutputMaybeMarker, newVarsMap);
        DumpIR(new IR.Tuple(newPrevCalls), "newPrevCalls");

        var newVarsMapFlatten = newVarsMap.SelectMany(x => x).ToArray();
        var newBody = MakeNewBody(fusion, newVarsMapFlatten.Select(v => v.InputIndex).ToHashSet().ToArray(), newPrevCalls);
        DumpIR(newBody, "newBody");
        var newParams = MakeNewParam(fusion, newVarsMapFlatten, newBody).ToHashSet().ToArray();
        var newFusion = fusion.With(body: newBody, parameters: newParams);
        var newArgs = MakeNewArgs(fusionOuterCall, newVarsMapFlatten, inputShouldBeMerge).ToHashSet().ToArray();

        Expr call = MakeNewCall(fusionOuterCall, fusion, newFusion, newArgs);

        // fusion var to arg
        // 左边的arg的表达式是右边arg的一部分的时候，在将左边的arg替换为var的时候
        // 右边的表达式中引用左边的arg的情况下右边的表达式也会被替换为fusion的var，参考TestMalMulReshape
        call = newParams.Zip(newArgs).Aggregate(call, (sum, pair) =>
        {
            var (param, arg) = pair;
            return ReplaceExpr(sum, param, arg);
        });

        DumpIR(call, $"{Counter++}_{_prevCallStr}_{fusion.Name}_after");
        ArgsChecker(newArgs);
        return call;
    }

    internal static T[] FusionVarsOperation<T>(VarReplaceInfo[] newVars, T[] fusionVars, Func<VarReplaceInfo, T[]> f)
        where T : Expr
    {
        var inputIndices = newVars.Select(v => v.InputIndex).ToArray();
        return fusionVars.ToArray().SelectMany((fusionArg, inputIndex) =>
        {
            // no change
            if (!inputIndices.Contains(inputIndex))
            {
                return new[] { fusionArg };
            }

            return newVars.Where(v => v.InputIndex == inputIndex).SelectMany(f).ToArray();
        }).ToArray();
    }

    internal static VarReplaceInfo[][] NewVarsDeduplication(VarReplaceInfo[][] newVars, Dictionary<Expr, Var> fusionDict)
    {
        var dict = newVars
            .SelectMany(x => x)
            .Select(info => new KeyValuePair<Expr, Var[]>(info.Expr, info.Vars))
            .ToHashSet(new KeyValuePairKeyComparer())
            .ToDictionary(pair => pair.Key, pair => pair.Value);
        var newVarsDeduplication = newVars.Select(list => list.Select(info =>
        {
            // todo: tuple的情况消除重复
            var defaultVar = Array.Empty<Var>();
            if (info.Expr is IR.Tuple tuple)
            {
                // FusionArg为tuple的时候，tuple中部分参数已经是fusion的参数的情况
                // TestMergeInputWhichHadBeMerged
                // TestMergeInputInTupleWhichHadBeMerged
                var callFields = tuple.Fields.ToArray().Where(ShouldBeInput).ToArray();

                // dict里有这个expr，也就是说其他FusionArg中出现过，有对应的vars
                if (dict.TryGetValue(info.Expr, out defaultVar))
                {
                    if (defaultVar.Length != callFields.Length)
                    {
                        throw new InvalidOperationException();
                    }
                }
                else
                {
                    // dict中没有这个var，那么只需要info中的vars是否有在fusionVar中出现过的
                    defaultVar = info.Vars;
                }

                var newVars = callFields.Zip(defaultVar).Select(pair =>
                {
                    // 如果tuple中有的元素已经在FusionArg中，那么优先替换
                    var (field, defaultVar) = pair;
                    if (fusionDict.TryGetValue(field, out var fusionVar))
                    {
                        return fusionVar;
                    }

                    // 否则使用默认的var todo: 添加这种test
                    return defaultVar;
                }).ToArray();
                return info with { Vars = newVars };
            }

            if (fusionDict.TryGetValue(info.Expr, out var fusionVar))
            {
                return info with { Vars = new[] { fusionVar } };
            }

            // TestSameInputMerge
            if (dict.TryGetValue(info.Expr, out var vars))
            {
                return info with { Vars = vars };
            }

            return info;
        }).ToArray()).ToArray();
        return newVarsDeduplication;
    }

    private static Expr[] MakeNewPrevCalls(Call[] inputsShouldBeMerge, Expr[] prevOutputMaybeMarker, VarReplaceInfo[][] newVarsOrigin)
    {
        var tuple = new IR.Tuple(inputsShouldBeMerge);

        if (inputsShouldBeMerge.Length != newVarsOrigin.Length)
        {
            Console.WriteLine();
        }

        return inputsShouldBeMerge.Zip(newVarsOrigin).Select((pair, i) =>
        {
            var (input, varsInfoList) = pair;
            int outCounter = 0;
            var newArgs = input.Arguments.ToArray().Select(x =>
            {
                if (x is TensorConst)
                {
                    return x;
                }

                if (x is Marker m && m.Target is TensorConst)
                {
                    return m;
                }

                if (outCounter >= varsInfoList.Length)
                {
                    throw new InvalidOperationException();
                }

                var newVar = varsInfoList[outCounter++].Vars;
                if (x is IR.Tuple tuple)
                {
                    int counter = 0;
                    var newFields = tuple.Fields.ToArray().Select(field =>
                    {
                        if (field is TensorConst)
                        {
                            return field;
                        }

                        return (Expr)newVar[counter++];
                    }).ToArray();
                    return new IR.Tuple(newFields);
                }

                return newVar.First();
            }).ToArray();
            var newCall = input.With(arguments: newArgs);

            var call = prevOutputMaybeMarker[i] is Marker m ? m.With(target: newCall) : (Expr)newCall;
            if (!call.InferenceType())
            {
                DumpIR(call, "InvalidInMakeNewPrevCalls");
                throw new InvalidOperationException();
            }

            return call;
        }).ToArray();
    }

    private static Call MakeNewCall(Call fusionOuterCall, BucketFusion fusion, BucketFusion newFusion, Expr[] newArgs)
    {
        // 原始的fusion的call更换target为新的fusion，以及arg0替换为prevCall的arg0，其他不变
        var call = fusionOuterCall.With(target: newFusion, arguments: newArgs);
        return call;
    }

    private static Expr[] MakeNewArgs(Call fusionOuterCall, VarReplaceInfo[] newVars, Call[] fusionArgs)
    {
        return FusionVarsOperation(newVars, fusionOuterCall.Arguments.ToArray(), newVar =>
        {
            if (newVar.Expr is IR.Tuple tuple)
            {
                return tuple.Fields.ToArray().Where(ShouldBeInput).ToArray();
            }

            return new[] { newVar.Expr };
        });
    }

    private static Var[] MakeNewParam(BucketFusion fusion, VarReplaceInfo[] newVars, Expr newBody)
    {
        var newParams = FusionVarsOperation(newVars, fusion.Parameters.ToArray(), newVar => newVar.Vars);
        return newParams;
    }

    private static Expr MakeNewBody(BucketFusion fusion, int[] inputIndices, Expr[] newPrevCalls)
    {
        // 新的fusion body将原来的var换成prevCall
        var newBody = inputIndices.Select(index => fusion.Parameters[index]).Zip(newPrevCalls).Aggregate(
            fusion.Body, (sum, pair) =>
            {
                // 此时prevCall携带新的var
                var (fusionVar, newPrevCall) = pair;
                return ReplaceExpr(sum, fusionVar, newPrevCall);
            });
        return newBody;
    }

    // todo: add test for this
    private static bool ShouldBeInput(Expr expr)
    {
        if (expr is Marker m)
        {
            return m.Target is not TensorConst;
        }

        return expr is not TensorConst;
    }

    // PrevCall(input1, input2, ...)
    // input: input1, input2, ...
    // call => [arg]
    // tuple => [arg1, arg2, ...]
    // VarReplaceInfo[InputIndex][InputArgIndex]
    private static VarReplaceInfo[][] MakeNewFusionVarsMap((Call, int)[] fusionInputsInfo, Dictionary<Expr, Var> fusionDict)
    {
        var newVars = fusionInputsInfo.Select(fusionInputInfo =>
        {
            var (fusionInput, inputIndex) = fusionInputInfo;
            return fusionInput.Arguments.ToArray().Where(ShouldBeInput).Select((inputArg) =>
            {
                // add condition to limit
                var vars = new[] { new Var(inputArg.CheckedType) };
                if (inputArg is IR.Tuple tuple)
                {
                    vars = tuple.Fields.ToArray().Where(ShouldBeInput).Select(field => new Var(field.CheckedType)).ToArray();
                }

                return new VarReplaceInfo(inputArg, vars, inputIndex);
            }).ToArray();
        }).ToArray();
        var newVarsDeduplication = NewVarsDeduplication(newVars, fusionDict);
        return newVarsDeduplication;
    }

    private Call[] CollectInputShouldBeMerge((Call, int)[] prevCallsInfo)
    {
        var prevCalls = prevCallsInfo.Select(x => x.Item1).ToArray();
        _prevCallStr = string.Join("_", prevCalls.Select(call => call.Target.GetType().Name));
        return prevCalls;
    }

    // 只需要替换被合并的call的args中的call，所以搜索和返回的都是Call
    // 记录index，和原始的call的arg对应，多个输入的情况可能中间会有const隔开
    private ((Call, int)[] PrevCalls, Expr[] MayBeMarkers) CollectInputsInfo(Call fusionOuterCall)
    {
        // todo: 判断rhs的effect var才行
        var prevCalls = new List<(Call, int)>();
        var maybeMarkers = new List<Expr>();
        var args = fusionOuterCall.Arguments.ToArray();
        for (int i = 0; i < args.Length; ++i)
        {
            var rhsArg = args[i];
            if (rhsArg is Marker marker && marker.Target is Call rhsPrevCall)
            {
                if (marker.Users.Count > 1)
                {
                    continue;
                }

                var rhsTarget = rhsPrevCall.Target;

                if (!IsInvalid(rhsPrevCall, rhsTarget))
                {
                    // prevCalls.Add((DupExpr(rhsPrevCall), i));
                    if (rhsPrevCall.CheckedType is TupleType)
                    {
                        Console.WriteLine("1065 Error");
                        throw new NotImplementedException();
                    }

                    prevCalls.Add((rhsPrevCall, i));
                    maybeMarkers.Add(marker);
                }
            }

            if (rhsArg is Call rhsCall)
            {
                var rhsTarget = rhsCall.Target;

                if (!IsInvalid(rhsCall, rhsTarget))
                {
                    if (rhsCall.CheckedType is TupleType)
                    {
                        Console.WriteLine("1080 Error");
                        throw new NotImplementedException();
                    }

                    // var rhs = DupExpr(rhsCall);
                    prevCalls.Add((rhsCall, i));
                    maybeMarkers.Add((Expr)rhsCall);
                }
            }
        }

        return (prevCalls.ToArray(), maybeMarkers.ToArray());
    }

    private bool IsInvalid(Call lhsPrevCall, Expr lhsTarget)
    {
        if (lhsPrevCall.Users.Count > 1)
        {
            return true;
        }

        if (!ValidTarget(lhsTarget))
        {
            return true;
        }

        return false;
    }
}

internal record VarReplaceInfo(Expr Expr, Var[] Vars, int InputIndex);
