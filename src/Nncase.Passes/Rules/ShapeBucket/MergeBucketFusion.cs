// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Reactive;
using System.Threading.Tasks;
using DryIoc.FastExpressionCompiler.LightExpression;
using DryIoc.ImTools;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Toolkit.HighPerformance;
using NetFabric.Hyperlinq;
using Nncase.IR;
using Nncase.IR.Tensors;
using Nncase.Passes.Analysis;
using Nncase.Passes.Mutators;
using Nncase.PatternMatch;
using static Nncase.Passes.Rules.ShapeBucket.ShapeBucketHelper;
using static Nncase.PatternMatch.Utility;
using static Nncase.Utilities.ReplaceUtility;
using Tuple = Nncase.IR.Tuple;

namespace Nncase.Passes.Rules.ShapeBucket;

[RuleGenerator]
public partial class MergeTupleFusion : RewriteRule<Pattern>
{
    public override Pattern Pattern => IsTuple(
        "tuple",
        new VArgsPattern(
            list =>
            {
                return Enumerable.Range(0, list.Length).Select(_ =>
                    IsWildcard(null, field => field is Call { Target: BucketFusion } && field.Users.Count == 1)).ToArray();
            },
            null));

    private Expr? GetReplace(Tuple tuple)
    {
        var fields = tuple.Fields.ToArray().OfType<Call>().ToArray();

        // merge input
        var newArgs = new List<Expr>();
        var newParams = new List<Var>();
        var oldParamsToNewArg = new List<(Var, Expr)>();
        foreach (var field in fields)
        {
            var fieldArgs = field.Arguments.ToArray();
            var fieldParams = ((BucketFusion)field.Target).Parameters;
            for (var i = 0; i < fieldArgs.Length; i++)
            {
                var fieldArg = fieldArgs[i];
                if (!newArgs.Contains(fieldArg))
                {
                    var newVar = new Var(fieldArg.CheckedType);
                    newParams.Add(newVar);
                    newArgs.Add(fieldArg);
                    oldParamsToNewArg.Add((fieldParams[i], newVar));
                }

                oldParamsToNewArg.Add((fieldParams[i], newParams[newArgs.IndexOf(fieldArg)]));
            }
        }

        var fieldBodys = fields.Select(c => c.Target).OfType<BucketFusion>().Select(x => x.Body).ToArray();
        var newBody = ReplaceClone(new IR.Tuple(fieldBodys), oldParamsToNewArg.ToArray());
        var newFusion = new BucketFusion("stackvm", newBody, newParams.ToArray(), Array.Empty<Var>());
        var newCall = new Call(newFusion, newArgs.ToArray());
        return newCall;
    }
}

public class MergeBucketFusion : ModulePass
{
    private static int _counter;

    private static string MergeRelPath => _counter.ToString();

    protected override Task<IRModule> RunCoreAsync(IRModule input, RunPassContext context)
    {
        // 1. save effect var info
        var main = (Function)input.Entry!;

        // var post = MergePrevFusion(main, set);
        // MergeMultiUsers(post);
        // return Task.FromResult(input);
        var hashcode = main.GetHashCode();
        int loop = 0;
        while (true)
        {
            var mergePrevPost = MergePrevFusion(main);
            MergeMultiUsers(mergePrevPost);
            MergeTupleFusion(mergePrevPost);
            var post = MergeMultiUsersSingleCall(mergePrevPost);
            var postHashCode = post.GetHashCode();
            if (hashcode != postHashCode)
            {
                _counter++;
            }
            else
            {
                break;
            }

            CheckErrorVar(post, main.Parameters.ToArray());
            CheckRepeat(post);
            hashcode = postHashCode;
            loop++;
        }

        return Task.FromResult(input);
    }

    private static void MergeTupleFusion(Function mergePrevPost) => CompilerServices.Rewrite(mergePrevPost, new[] { new MergeTupleFusion() }, new());

    private static void MergeMultiUsers(Function post)
    {
        IRHelpers.DCE(post);
        DumpIR(post, "AfterDCE", MergeRelPath);
        var c = new ReplaceVisitor();
        c.Replace(post);
        DumpIR(post, "AfterMergeUser", MergeRelPath);
    }

    private static void CheckRepeat(Expr call)
    {
        // todo: 检查所有fusion里面的param有没有重复名字的
        // todo: 检查有没有fusion名字重复的
        var c = new CheckFusionCallVisitor();
        c.Visit(call);
        c.Check();
    }

    private static void CheckErrorVar(Expr body, Var[] vars)
    {
        var f = new FindVar();
        f.Visit(body);
        if (!f.Vars.All(vars.Contains))
        {
            Console.WriteLine(string.Join(", ", f.Vars.Select(x => x.Name).ToArray()));
            throw new InvalidOperationException("Has Invalid Var In Body");
        }
    }

    private static bool DetectedRing(Call outerCall, Expr[] users)
    {
        // var users = outerCall.Users.ToArray();
        var userArgs = users.SelectMany(user => ((Call)user).Arguments.ToArray()).ToArray();
        foreach (var arg in userArgs)
        {
            var list = new FindExpr().Run(arg, users, outerCall, expr =>
            {
                if (expr is Const)
                {
                    return false;
                }

                return users.Contains(expr);
            });
            if (list.Count > 0)
            {
                return true;
            }
        }

        return false;
    }

    private static (Expr? NewCall, Expr[] AllUsers) MergeMultiUserFusion(Call outerCall, BucketFusion fusion)
    {
        var users = outerCall.Users.ToArray();

        if (users.Length == 0)
        {
            return (null, Array.Empty<Expr>());
        }

        if (users.OfType<Call>().All(user => user.Target is GetItem))
        {
            // todo: test
            // 需要去重，可能一个getItem的user使用了多个getItem
            // 但是去重的过程需要在collect info的时候做
            // 不然被去重的user不能正确的被替换掉
            users = users.SelectMany(user => user.Users.ToArray()).ToArray();
        }

        // todo: not support
        if (users.Any(user => user is Tuple))
        {
            Console.WriteLine("HasTuple");
            return (null, Array.Empty<Expr>());
        }

        // todo:如果不是所有的都是valid的，那么能合并吗？？
        var userInfos = CollectUsers(outerCall, users);

        // todo: support only one user, because merge fusion rule is not enough
        // maybe a error
        // if (userInfos.Length < 2)
        // {
        //     return null;
        // }

        // has invalid
        if (userInfos.Length != users.Distinct().ToArray().Length)
        {
            Console.WriteLine("not all call");
            return (null, Array.Empty<Expr>());
        }

        if (outerCall.Users.Any(user => user is Tuple) || users.Any(user => user.CheckedType is TupleType))
        {
            return (null, Array.Empty<Expr>());
        }

        if (users.Any(user =>
                user is Call c && c.Arguments.ToArray().Any(arg => arg is Tuple || arg.CheckedType is TupleType)))
        {
            // todo: not implement
            return (null, Array.Empty<Expr>());
        }

        if (DetectedRing(outerCall, users))
        {
            Console.WriteLine("HasRing");
            return (null, Array.Empty<Expr>());
        }

        if (outerCall.Users.ToArray().OfType<Call>().All(user => user.Target is GetItem))
        {
            Console.WriteLine("MeregForGetItem");
            Console.WriteLine(MergeRelPath);
        }

        // todo: with tuple
        // 1. all user are fusion
        // 2. some user are fusion
        // 3. no fusion
        var oldUsers = userInfos.Select(x => x.User).ToArray();

        var otherName = string.Join("\n", oldUsers.Select(x => x.Target switch
        {
            BucketFusion f => f.Name,
            Op op => op.GetType().Name,
            _ => string.Empty,
        }));
        Console.WriteLine($"Merge {fusion.Name}");
        Console.WriteLine(otherName);
        var fusionDict = outerCall.Arguments.ToArray().Zip(fusion.Parameters.ToArray()).ToArray();

        // 这个vars用于确定output的args里面哪些要加入，哪些要消除，另外还要包含多个user的那个
        // todo: 目前newParams已经去除重复
        var argMap = MakeNewVarsMap(userInfos, fusionDict, outerCall);
        var newUsers = MakeNewUserExpr(userInfos, outerCall, argMap);
        var newBody = MakeNewBody(newUsers);

        // todo: args去除重复，在不需要更新的情况下不进行更新
        var newArgs = argMap.NewArgs();
        var newParams = argMap.NewParams;
        var newFusion = MakeNewFusion(newBody, fusion, newParams, oldUsers);
        var newCall = MakeNewCall(newFusion, newArgs);
        if (newArgs.ToHashSet().Count != newArgs.Length)
        {
            throw new InvalidOperationException("Has Repeat args");
        }

        if (!newCall.InferenceType())
        {
            DumpIR(newCall, "newCallInvalid");
            throw new InvalidOperationException("InvalidNewCallInMergeMultiUser");
        }

        DumpIR(newCall, "newCall", MergeRelPath);
        ArgsChecker(newArgs);
        return (newCall, users);
    }

    private static FusionVarMapper MakeNewVarsMap(UserInfo[] userInfos, (Expr, Var)[] fusionDict, Call outerCall)
    {
        var originUsers = outerCall.Users.ToArray();
        var originArgs = fusionDict.Concat(userInfos.SelectMany(info =>
        {
            var user = info.User;
            if (user.Target is BucketFusion fusion)
            {
                if (info.GetItem != null)
                {
                    var args = user.Arguments.ToArray().OfNoConst().ToArray();

                    // 这里需要删除所有outerCall user里面的getItem，有可能一个call用了多个getItem
                    var index = args.IndexOf(info.GetItem);
                    return args.Zip(fusion.Parameters.ToArray()).Where(pair => !originUsers.Contains(pair.First)).ToArray();
                }

                return user.Arguments.ToArray().OfNoConst().Zip(fusion.Parameters.ToArray());
            }

            throw new NotImplementedException();
        })).ToArray();

        var oldVars = originArgs.Select(item => item.Item2).ToArray();
        if (oldVars.ToHashSet().Count != oldVars.Length)
        {
            throw new InvalidOperationException("oldVar has dup");
        }

        var users = userInfos.Select(x => x.User).ToArray();

        // 保证var顺序以及字典
        // 初始param应该包含fusion的
        var fusionParams = fusionDict.Select(pair => pair.Item2).ToArray();

        // arg到var的映射，并不需要使用fusion的信息设置为初始值，因为合并的是user
        var result = originArgs.Aggregate(
            (Array.Empty<(Expr UserArg, Var RelativeNewVar, Var OldVar)>(), fusionParams),
            (sum, pair) =>
            {
                var (totalDict, totalVars) = sum;
                var (arg, oldVar) = pair;
                var fusionResult = FindFirst(fusionDict, arg);

                // fusion的参数中查看是否有这个arg,有的话则用fusion对应的var来替代
                if (fusionResult != null)
                {
                    // 这个时候会有多个arg指向同一个RelativeNewVar，但是为了把这个arg替换为oldvar, 暂时要保留，后面需要去重
                    // 不能在这里就去重，会引发错误
                    return (totalDict.Append((arg, fusionResult!, oldVar)), totalVars);
                }

                // 查看当前累积的参数中是否有这个arg,有的话用arg对应的新var来替代
                var result = FindFirst(totalDict.Select(tuple => (tuple.UserArg, tuple.RelativeNewVar)).ToArray(), arg);
                if (result != null)
                {
                    return (totalDict.Append((arg, result!, oldVar)), totalVars);
                }

                // todo: maybe put fusion body in this is better?
                // arg是outerCall那就不需要添加任何var，因为这个arg会被替换为fusion的body
                if (arg == outerCall)
                {
                    return (totalDict, totalVars);
                }

                // 其他参数被合并进来了，也就不需要再创建新的var,但是后面替换的时候也要对这种情况进行替换
                if (users.Contains(arg))
                {
                    return (totalDict, totalVars);
                }

                // 普通的其他输入，进行替换
                var newVar = new Var(arg.CheckedType);
                return (totalDict.Append((arg, newVar, oldVar)), totalVars.Append(newVar));
            });
        return new FusionVarMapper(result.fusionParams, result.Item1);
    }

    private static Var? FindFirst((Expr, Var)[] totalDict, Expr arg)
    {
        var result = totalDict.Where(pair => pair.Item1 == arg).ToArray();
        if (result.Length > 0)
        {
            return result.First().Item2;
        }

        return null;
    }

    private static BucketFusion MakeNewFusion(Expr body, BucketFusion fusion, Var[] newParams, Call[] oldUsers)
    {
        // todo: EffectVar
        var name = fusion.Name + "_" + string.Join("_", oldUsers.Select(x => x.Target switch
        {
            BucketFusion f => f.Name,
            Op op => op.GetType().Name,
            _ => string.Empty,
        }));
        if (name.Length > 100)
        {
            name = name.Substring(0, 100);
        }

        return new BucketFusion(name, "stackvm", body, newParams, fusion.EffectVar);
    }

    private static Call MakeNewCall(BucketFusion fusion, Expr[] args)
    {
        return new Call(fusion, args);
    }

    private static Expr[] MakeNewArgs(Call outerCall, UserInfo[] userInfos, VarReplInfo[] newVarsMap)
    {
        var fusionArgs = outerCall.Arguments.ToArray();
        var newArgs = userInfos.SelectMany(userInfo =>
        {
            var user = userInfo.User;
            return user.Arguments.ToArray().Remove(outerCall).OfNoConst().ToArray();
        }).ToHashSet().ToArray();
        return fusionArgs.Concat(newArgs).ToArray();
    }

    // private static Var[] MakeNewParams(BucketFusion fusion, UserInfo[] userInfos, VarReplInfo[] newVarsMap)
    // {
    // return newVarsMap.SelectMany(newVar => newVar.Vars).ToArray();
    // var fusionParams = fusion.Parameters.ToArray();
    // var newParams = userInfos.SelectMany(userInfo =>
    // {
    // todo: process for Fusion
    // var usersArgs = userInfo.User.Arguments.ToArray().RemoveAt(userInfo.FusionIndexInUserArg).OfNoConst().ToArray();
    // return usersArgs.Select(arg => new Var(arg.CheckedType)).ToArray();
    // }).ToHashSet().ToArray();
    // }
    private static Expr MakeNewBody(Expr[] newUsers)
    {
        if (newUsers.Length == 1)
        {
            return newUsers.First();
        }

        return new IR.Tuple(newUsers);
    }

    private static Expr[] MakeNewUserExpr(UserInfo[] userInfos, Call outerCall, FusionVarMapper argMap)
    {
        var users = userInfos.Select(x => x.User).ToArray();

        // 不能加入原始的body，因为所有的输出都被合并了，只有合并进来的输出了
        // todo: 如果user是getItem，那么需要换成getItem的target才行
        var originBody = new IR.Tuple(users.ToArray());

        // var newOriginBody =
        // new IR.Tuple(new Expr[] { body }
        // .Concat(users.Select(user => user.Target).OfType<BucketFusion>().Select(f => f.Body)).ToArray());
        var newOriginBody = originBody.Clone();
        var finder = new FindVar();
        finder.Visit(newOriginBody);
        var newVars = finder.Vars;

        // todo: this is unused
        var map = argMap.OldToNewParam();

        // replace
        var fusionMap = newOriginBody.Fields.ToArray().Append(outerCall).OfType<Call>().Where(call => call.Target is BucketFusion)
            .ToDictionary(call => call.Target, call => ((BucketFusion)call.Target).Body);
        var mutator = new Passes.Mutators.Substitutor(e =>
        {
            // 因为替换一个以后其他的都会发生变化，因此不能用call来匹配
            if (e is Call c)
            {
                if (c.Target is not BucketFusion)
                {
                    return null;
                }

                // 只有在call是属于users中才行，否则是
                var fusion = (BucketFusion)c.Target;
                if (fusionMap.TryGetValue(fusion, out var body))
                {
                    // 被合并的fusion不需要做额外的操作
                    if (fusion.Name == ((BucketFusion)outerCall.Target).Name)
                    {
                        return body;
                    }

                    var paramList = fusion.Parameters.ToArray();

                    // todo: 注意，如果arg target不是users里面的fusion，那么就不替换。目前没实现这个部分
                    var pairs = paramList.Zip(c.Arguments.ToArray()).Select((pair) =>
                    {
                        // 默认是替换为arg,但是如果有其他可以替换的，那么替换为其他的
                        var (param, arg) = pair;
                        if (map.TryGetValue(param, out var newVar))
                        {
                            return (param, newVar);
                        }

                        return pair;
                    });

                    // body中的var需要换成call对应的arg
                    return pairs.Aggregate(body, (sum, pair) =>
                    {
                        return ReplaceExpr(sum, pair.First, pair.Second);
                    });

                    // return body;
                }
            }

            return null;
        });
        mutator.Visit(newOriginBody, Unit.Default);

        // DumpIR(newOriginBody, "newOriginBody", relPath);
        return newOriginBody.Fields.ToArray();
    }

    private static (Var, Expr)[] MakeFusionReplaceInfo(UserInfo[] userInfos, Expr body, Call[] users, Dictionary<BucketFusion, Var> fusionMap) =>
        userInfos.SelectMany((userInfo, i) =>
        {
            var user = userInfo.User;

            // 取出user的body
            if (user.Target is BucketFusion fusion)
            {
                var replaceInfo = fusion.Parameters.ToArray().Zip(user.Arguments.ToArray()).Select((pair, i) =>
                {
                    var (param, arg) = pair;

                    // arg是其他的fusion的话替换为其fusion的body，但是其中的var可能要后面才能替换了
                    if (users.Contains(arg))
                    {
                        if (((Call)arg).Target is BucketFusion argFusion)
                        {
                            return (param, argFusion.Body);
                        }
                        else
                        {
                            throw new NotImplementedException();
                        }
                    }

                    return pair;
                }).ToArray();
                return replaceInfo;
            }

            throw new NotImplementedException();
        }).ToArray();

    private static UserInfo[] CollectUsers(Call outerCall, Expr[] users)
    {
        var originUsers = outerCall.Users.ToArray();
        var outputs = users
            .Distinct() // 去重，避免多个getItem被一个expr使用
            .Select((user, userIndex) =>
            {
                // 找到fusion在user的arguments的哪个index
                if (user is Call userCall)
                {
                    var valid = userCall.Target switch
                    {
                        Op op => false,
                        BucketFusion => true, // todo: check effect var
                        _ => throw new NotImplementedException(),
                    };
                    return (user, userIndex, valid);
                }
                else if (user is IR.Tuple)
                {
                    // tuple should in other merge rule
                    var valid = false;
                    return (user, userIndex, valid);
                }
                else
                {
                    var valid = false;
                    return (user, userIndex, valid);
                }
            })
            .Where(tuple => tuple.valid)
            .Select(tuple => (tuple.user, tuple.userIndex))
            .Select(pair =>
            {
                var user = (Call)pair.user;
                var getItem = user.Arguments.ToArray().FindFirst(userArg => originUsers.Contains(userArg));
                return new UserInfo(user, pair.userIndex, getItem);
            })
            .ToArray();
        return outputs;
    }

    private static void TranslateFusionToBucket(Dictionary<string, Var[]> set, Function post, CompileSession seesion)
    {
        var inputDimsVars = InputDimVars(seesion);
        var mutator = new Passes.Mutators.Substitutor(e =>
        {
            if (e is Call c && c.Target is Fusion f)
            {
                // CompilerServices.Rewrite(f.Body, new[] { new FoldRepeatMarker() }, new());
                var effectVars = Array.Empty<Var>();
                if (inputDimsVars.Length <= 1)
                {
                    effectVars = inputDimsVars;
                }
                else
                {
                    effectVars = f.Name.Split("_").Chunk(2).SelectMany(list =>
                    {
                        var originName = string.Join("_", list);
                        return set[originName];
                    }).ToHashSet().ToArray();
                }

                return c.With(target: BucketFusion.FromNormalFusion(f, effectVars));
            }

            return null;
        });
        mutator.Visit(post, Unit.Default);
    }

    private Expr MergeMultiUsersSingleCall(Expr body)
    {
        return CompilerServices.Rewrite(body, new IRewriteRule[] { new MultiUserCallToFusion() }, new());
    }

    private Function MergePrevFusion(Function main)
    {
        // 1. get origin info
        var s = new SearchBucketFusion();
        s.Visit(main);
        var set = s.FusionEffectVars();

        // 2. merge
        var post = MergeFusion(main);
        DumpIR(post, "AfterMergeFusion", MergeRelPath);

        // 3. translate fusion to BucketFusion
        TranslateFusionToBucket(set, post, CompileSession);
        DumpIR(post, "AfterTranslateFusion", MergeRelPath);
        return post;
    }

    private Function MergeFusion(Function main)
    {
        var analyzerMananger = CompileSession.GetRequiredService<IAnalyzerManager>();
        var analysis = new Dictionary<Type, IAnalysisResult>
        {
            [typeof(IExprUserAnalysisResult)] = analyzerMananger.GetAnaylsis<IExprUserAnalysisResult>(main),
        };
        CompilerServices.Rewrite(main, new[] { new ClearFusionOuterMarker() }, new());
        var rewriter = new DataFlowMergeRewriter();
        var post = (Function)rewriter.Rewrite(
            main,
            new IMergeRewriteRule[]
            {
                new SameInputFusionMergeRule(), new MultiInputFusionMergeRule(), new ShortCutFusionMergeRuleLeft(),
                new ShortCutFusionMergeRuleRight(),
            },
            (rule, option) => new BucketFusionGroupMutator(rule, option),
            new() { AnalysisResults = analysis });

        return post;
    }

    private record UserInfo(Call User, int UserIndex, Expr? GetItem);

    private class ReplaceVisitor : ExprVisitor<Expr, Unit>
    {
        private Function? _fn;

        private bool _changed;

        private static int Counter => MultiUserCallToFusion.Counter;

        private static string RelPath => Counter.ToString();

        private Expr Root => _fn!.Body;

        public void Replace(Function fn)
        {
            _fn = fn;
            Visit(Root);
        }

        protected override Expr VisitCall(Call expr)
        {
            if (_changed)
            {
                return expr;
            }

            return base.VisitCall(expr);
        }

        protected override Expr VisitLeafCall(Call expr)
        {
            // 回避错误的user计数与visitor的访问
            // visitor的逻辑会先访问叶子节点
            // 因此如果VisitLeaf叶子改了其user
            // 那么expr同级的其他operand即便被修改了也还是会访问到
            // 因此一次只能做一个
            if (_changed)
            {
                return expr;
            }

            if (expr is Call outerCall && outerCall.Target is BucketFusion fusion)
            {
                // Console.WriteLine($"Match {fusion.Name} counter:{Counter}");
                DumpIR(Root, "OriginRoot", RelPath);

                var (newCall, users) = MergeMultiUserFusion(outerCall, fusion);
                if (newCall != null)
                {
                    UpdateUse(users, newCall, outerCall);
                    DumpIR(Root, "rootAfterMerge", RelPath);
                    Root.InferenceType();
                    if (Root.CheckedType is InvalidType)
                    {
                        throw new InvalidOperationException("InvalidRoot");
                    }

                    // todo: 检查已经被合并的fusion的名字是否还存在，存在就是错误
                    AddCounter();
                    CheckRepeat(Root);
                    _changed = true;
                    return newCall;
                }
            }

            return expr;
        }

        protected override Expr DefaultVisitLeaf(Expr expr) => expr;

        private static void UpdateUse(Expr[] users, Expr newCall, Call outerCall)
        {
            // ref TestTupleGetItemOutputIsSingle
            if (users.Distinct().ToArray().Length == 1)
            {
                ReplaceAllUsesWith(users[0], newCall);
                return;
            }

            var originUsersIndex = 0;
            var getItemMode = outerCall.Users.First() is Call c && c.Target is GetItem;
            if (getItemMode)
            {
                // 第几个GetItem对应的users用同一个operand
                for (int i = 0; i < outerCall.Users.Count; i++)
                {
                    var newOperand = newCall[i];
                    for (int j = 0; j < outerCall.Users.ToArray()[i].Users.Count; j++)
                    {
                        ReplaceAllUsesWith(users[originUsersIndex], newOperand);
                        originUsersIndex++;
                    }
                }
            }
            else
            {
                for (var i = 0; i < users.Length; i++)
                {
                    var newOperand = newCall.CheckedType is TupleType ? newCall[i] : newCall;
                    ReplaceAllUsesWith(users[i], newOperand);
                }
            }
        }

        private void AddCounter()
        {
            MultiUserCallToFusion.Counter++;
        }
    }
}

internal sealed class BucketFusionGroupMutator : Passes.Mutators.FusionGroupMutator
{
    public BucketFusionGroupMutator(IMergeRewriteRule preOrderfusionRule, RunPassContext passOptions)
        : base(preOrderfusionRule, passOptions)
    {
    }

    public override bool MergedFusionCheckCallBack(Fusion mergedFusion, HashSet<Fusion> candidateFusions)
    {
        // ShowMergeCandidate(mergedFusion, candidateFusions);

        // 回避反卷积，反卷积的shape表达式目前会引起重复的计算
        if (mergedFusion.Name.Contains("Conv2DTranspose", StringComparison.Ordinal) ||
            candidateFusions.Any(f => f.Name.Contains("Conv2DTranspose", StringComparison.Ordinal)))
        {
            return false;
        }

        return true;
    }

    private static void ShowMergeCandidate(Fusion mergedFusion, HashSet<Fusion> candidateFusions)
    {
        Console.WriteLine("-----------------");
        Console.WriteLine(mergedFusion.Name);
        Console.WriteLine("-----------------");
        foreach (var candidateFusion in candidateFusions)
        {
            Console.WriteLine(candidateFusion.Name);
        }

        Console.WriteLine("-----------------");
    }
}

// params和args就通过这个来构建
// 通过引用的arg来判断是否重复，先有args,再生成对应的params，index已经不重要了
internal record VarReplInfo(Var[] Vars, Expr[] Exprs);

// userArg -> oldVar 原始fusion的var
// oldVar -> RelativeNewVar 替换的过程
internal record FusionVarMapper(Var[] NewParams, (Expr UserArg, Var RelativeNewVar, Var OldVar)[] ArgMap)
{
    public Dictionary<Var, Var> OldToNewParam()
    {
        return ArgMap.ToDictionary(info => info.OldVar, info => info.RelativeNewVar);
    }

    public Expr[] NewArgs()
    {
        // 多个arg指向相同的RelativeNewVar的情况，去重
        var data = ArgMap.Where(pair => NewParams.Contains(pair.RelativeNewVar)).DistinctBy(x => x.RelativeNewVar).ToArray();
        if (data.Length != NewParams.Length)
        {
            Console.WriteLine("error");
        }

        return data.Select(pair => pair.UserArg).ToArray();
    }
}

internal sealed class CheckFusionCallVisitor : ExprWalker
{
    private HashSet<string> _callName = new();
    private Dictionary<string, (string, BucketFusion)> _errorFusion = new();

    private HashSet<string> _fusionName = new();
    private HashSet<string> _repeatFusion = new();

    private HashSet<string> _fusionParamsName = new();
    private HashSet<string> _repeatParamFusion = new();

    public void Check()
    {
        var error = false;
        if (_errorFusion.Count != 0)
        {
            error = true;
            Console.WriteLine("errorFusion");
        }

        if (_repeatFusion.Count != 0)
        {
            error = true;
            Print("repeatFusion not zero", _repeatFusion);
        }

        if (_repeatParamFusion.Count != 0)
        {
            error = true;
            Print("repeatParamFusion not zero", _repeatParamFusion);
        }

        if (error)
        {
            throw new InvalidOperationException();
        }
    }

    protected override Unit VisitLeafFusion(Fusion fusion)
    {
        // 可能有多个user啊，每次进来访问
        if (fusion is BucketFusion bf)
        {
            if (_fusionName.Contains(bf.Name))
            {
                _repeatFusion.Add(bf.Name);
            }
            else
            {
                _fusionName.Add(bf.Name);
            }

            var parameters = bf.Parameters.ToArray();
            foreach (var parameter in parameters)
            {
                if (_fusionParamsName.Contains(parameter.Name))
                {
                    _repeatParamFusion.Add(parameter.Name);
                }
            }

            _fusionParamsName.UnionWith(parameters.Select(p => p.Name).ToArray());
        }

        return default;
    }

    private void Print(string name, HashSet<string> list)
    {
        Console.WriteLine(name);
        foreach (string s in list)
        {
            Console.WriteLine(s);
        }
    }
}

internal class SearchBucketFusion : ExprVisitor<Expr, Unit>
{
    private HashSet<BucketFusion> FusionSet { get; set; } = new();

    public Dictionary<string, Var[]> FusionEffectVars()
    {
        return FusionSet.ToDictionary(s => s.Name, s => s.EffectVar);
    }

    protected override Expr DefaultVisitLeaf(Expr expr) => expr;

    protected override Expr VisitLeafCall(Call expr)
    {
        if (expr.Target is BucketFusion f)
        {
            FusionSet.Add(f);
        }

        return expr;
    }
}
