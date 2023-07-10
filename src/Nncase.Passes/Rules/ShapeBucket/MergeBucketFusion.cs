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
using Nncase.Passes.Analysis;
using Nncase.Passes.Mutators;
using static Nncase.Utilities.ReplaceUtility;
using static Nncase.Passes.Rules.ShapeBucket.ShapeBucketHelper;

namespace Nncase.Passes.Rules.ShapeBucket;

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

public class MergeBucketFusion : ModulePass
{
    protected override Task<IRModule> RunCoreAsync(IRModule input, RunPassContext context)
    {
        // 1. save effect var info
        var main = (Function)input.Entry!;
        var s = new SearchBucketFusion();
        s.Visit(main);
        var set = s.FusionEffectVars();

        // 2. merge
        var post = MergeFusion(main);
        DumpIR(post, "AfterMergeFusion");

        // 3. translate fusion to BucketFusion
        TranslateFusionToBucket(set, post);
        DumpIR(post, "AfterTranslateFusion");

        IRHelpers.DCE(post);
        var c = new ReplaceVisitor();
        c.Replace(post);
        return Task.FromResult(input);
    }

    record UserInfo(Call User, int UserIndex, int FusionIndexInUserArg);

    internal class ReplaceVisitor : ExprVisitor<Expr, Unit>
    {
        private static int counter = 0;

        private Function _fn;
        private Expr _root => _fn.Body;

        public void Replace(Function fn)
        {
            _fn = fn;
            Visit(_root);
        }

        // todo: 问题
        // 1. getItem + getItem
        // 2. 如何合并getItem的多个users
        // 3. InputVar 去重？？？
        protected override Expr VisitLeafCall(Call expr)
        {
            var fusionPattern = new FusionBucket().Pattern;
            if (CompilerServices.TryMatchRoot(expr, fusionPattern, out var matchResult) && expr.Users.Count > 1)
            {
                var outerCall = (Call)matchResult["outerCall"];
                var fusion = (BucketFusion)matchResult["fusion"];
                DumpIR(outerCall, $"{counter}_MergeBefore");
                for (var i = 0; i < outerCall.Users.Count; i++)
                {
                    DumpIR(outerCall.Users.ToArray()[i], $"{counter}_User_{i}_MergeBefore");
                }
                var newCall = MergeMultiUserFusion(outerCall, fusion);
                if (newCall != null)
                {
                    DumpIR(newCall, $"{counter++}_MergeAfter");
                    DumpIR(_root, "rootBeforeMerge");
                    var users = outerCall.Users.ToArray();
                    for (var i = 0; i < users.Length; i++)
                    {
                        var newOperand = newCall.CheckedType is TupleType ? newCall[i] : newCall;
                        ReplaceAllUsesWith(users[i], newOperand);
                    }
                    // ReplaceAllUsesWith(outerCall, newCall);
                    DumpIR(_root, "rootAfterMerge");
                    return newCall;
                }
            }
            return expr;
        }

        protected override Expr DefaultVisitLeaf(Expr expr) => expr;
    }

    private static Expr? MergeMultiUserFusion(Call outerCall, BucketFusion fusion)
    {
        var userInfos = CollectUsers(outerCall);

        // maybe a error
        if (userInfos.Length < 2)
        {
            return null;
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
            _ => ""
        }));
        Console.WriteLine($"Merge {fusion.Name}");
        Console.WriteLine(otherName);
        var fusionDict = outerCall.Arguments.ToArray().Zip(fusion.Parameters.ToArray()).ToArray();
        // 这个vars用于确定output的args里面哪些要加入，哪些要消除，另外还要包含多个user的那个
        // todo: 目前newParams已经去除重复
        var (argMap, newParams) = MakeNewVarsMap(userInfos, fusionDict, outerCall);
        for (int i = 0; i < argMap.Length; i++)
        {
            Console.WriteLine(argMap[i].Item1.GetHashCode());
            Console.WriteLine(((Call)argMap[i].Item1).Target);
            Console.WriteLine(argMap[i].Item2.Name);
        }
        var newUsers = MakeNewUserExpr(userInfos, fusion.Body, argMap);
        var newBody = MakeNewBody(newUsers);
        DumpIR(newBody, "newBody");

        // todo: args去除重复，在不需要更新的情况下不进行更新
        var newArgs = argMap.Where(pair => newParams.Contains(pair.Item2)).Take(newParams.Length)
            .Select(pair => pair.Item1).ToArray();

        // var newParams = newArgs.Select(arg => new Var(arg.CheckedType)).ToArray();
        // var newParams = MakeNewParams(fusion, userInfos, newVarsMap);
        // var newArgs = MakeNewArgs(outerCall, userInfos, newVarsMap);
        var newFusion = MakeNewFusion(newBody, fusion, newParams, oldUsers);
        var newCall = MakeNewCall(newFusion, newArgs);
        return newCall;
    }

    record VarReplInfo(Var[] vars, Expr[] exprs);
    // params和args就通过这个来构建
    // 通过引用的arg来判断是否重复，先有args,再生成对应的params，index已经不重要了

    //

    // 所有新的param
    // 新的param与新的arg的对应关系
    // 所有fusion的var和新的var的对应关系 （用于替换fusion中的var） todo：
    record FusionVarMapper(Var[] NewParams, Dictionary<Var, Var> oldToNewParam, Dictionary<Var, Expr> paramToArg);

    private static ((Expr, Var)[] ArgMap, Var[] NewVars) MakeNewVarsMap(UserInfo[] userInfos, (Expr, Var)[] fusionDict, Call outerCall)
    {
        var originArgs = fusionDict.Select(pair => pair.Item1).Concat(userInfos.SelectMany(info =>
        {
            var user = info.User;
            return user.Arguments.ToArray().OfNoConst();
        })).ToArray();

        // 保证var顺序以及字典
        // 初始param应该包含fusion的
        var fusionParams = fusionDict.Select(pair => pair.Item2).ToArray();
        var result = originArgs.Aggregate((new (Expr, Var)[]{}, fusionParams), (sum, arg) =>
        {
            var (totalDict, totalVars) = sum;
            var fusionResult = FindFirst(fusionDict, arg);
            // todo: 这里没有判断成功
            if (fusionResult != null)
            {
                return (totalDict.Append((arg, fusionResult!)), totalVars);
            }

            var result = FindFirst(totalDict, arg);
            if (result != null)
            {
                return (totalDict.Append((arg, result!)), totalVars);
            }

            // todo: maybe put fusion body in this is better?
            if (arg == outerCall)
            {
                return (totalDict, totalVars);
            }

            var newVar = new Var(arg.CheckedType);
            return (totalDict.Append((arg, newVar)), totalVars.Append(newVar));
        });
        // // 要保留每个user里面的argument对应的是新fusion的哪一个var
        // var allArgs = userInfos.SelectMany(info =>
        // {
        //     var user = info.User;
        //     // todo: tuple处理
        //     var args = user.Arguments.ToArray().RemoveAt(info.FusionIndexInUserArg).OfNoConst();
        //     return args.ToArray();
        // }).ToHashSet().ToArray();
        // return allArgs;
        return result;
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
            _ => ""
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
            return user.Arguments.ToArray().RemoveAt(userInfo.FusionIndexInUserArg).OfNoConst().ToArray();
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

    // clone origin Expr and Do replace for var
    private static Expr ReplaceClone(Expr originBody, params (Var, Expr)[] originVarAndExpr)
    {
        DumpIR(originBody, "replace_begin");
        var call = originBody.Clone();
        DumpIR(call, "replace_clone");

        var finder = new FindVar();
        finder.Visit(call);
        var newVars = finder.Vars;
        originVarAndExpr.ForEach(pair =>
        {
            var (v, newExpr) = pair;
            var varShouldBeReplaced = newVars.FindFirst(newVar => newVar.Name == v.Name);
            ReplaceExpr(call, varShouldBeReplaced, newExpr);
        });
        DumpIR(originBody, "replace_end");
        return call;
    }

    private static Expr[] MakeNewUserExpr(UserInfo[] userInfos, Expr body, (Expr, Var)[] argMap)
    {
        // make user expr, replace call with fusion body
        return userInfos.Select((userInfo, i) =>
        {
            var user = userInfo.User;

            // 两部分
            // 被合并的fusion的body中，其中引用到fusion的部分的var替换为fusion的body 目前没有问题
            // 其他的var使用ReplaceInfo来替换， replaceInfo应该包括所有fusion的var，以及对应的表达式

            var argReplaceInfo = argMap.Select(pair => (pair.Item2, pair.Item1)).ToArray();
            Expr newUser;
            // maybe marker
            if (user.Target is BucketFusion fusion)
            {
                var replaceInfo = fusion.Parameters.ToArray().Zip(user.Arguments.ToArray()).Select((pair, i) =>
                {
                    // arg是fusion的话替换为fusion的body
                    var (param, arg) = pair;
                    if (i == userInfo.FusionIndexInUserArg)
                    {
                        return (param, body);
                    }
                    // 否则expr要换成新的var才行
                    var newVar = FindFirst(argMap, arg);
                    if (newVar != null)
                    {
                        return (param, newVar);
                    }

                    throw new NotImplementedException();
                }).ToArray();
                newUser = ReplaceClone(fusion.Body, replaceInfo);
            }
            else if (user.Target is Op op)
            {
                newUser = ReplaceCallParams(userInfo.User, (userInfo.FusionIndexInUserArg, body));
            }
            else
            {
                throw new NotImplementedException();
            }
            // Expr newUser = user.Target switch
            // {
            //     BucketFusion fusion => ReplaceClone(fusion.Body, argReplaceInfo.Append((fusion.Parameters.ToArray()[userInfo.FusionIndexInUserArg], body))),
            //     // todo: fix this;
            //     Op op => ReplaceCallParams(userInfo.User, (userInfo.FusionIndexInUserArg, body)),
            //     _ => throw new NotImplementedException(),
            // };

            DumpIR(newUser, $"newUser_{i}");
            return newUser;
            }).ToArray();
    }

    private static UserInfo[] CollectUsers(Expr outerCall)
    {
        var outputs = outerCall.Users
            .Select((user, userIndex) =>
            {
                // 找到fusion在user的arguments的哪个index
                if (user is Call userCall)
                {
                    var valid = userCall.Target switch
                    {
                        Op op => CallValidator.ValidTarget(op),
                        // todo: check effect var
                        BucketFusion => true,
                        // what this
                        _ => throw new NotImplementedException(),
                    };
                    return (user, userIndex, valid);
                }
                else
                {
                    throw new NotImplementedException();
                }
            })
            .Where(tuple => tuple.valid)
            .Select(tuple => (tuple.user, tuple.userIndex))
            .Select(pair =>
            {
                var user = (Call)pair.user;
                var argInUserArg = user.Arguments.ToArray().IndexOf(outerCall);
                return new UserInfo(user, pair.userIndex, argInUserArg);
            })
            .ToArray();
        return outputs;
    }

    private static void TranslateFusionToBucket(Dictionary<string, Var[]> set, Function post)
    {
        var mutator = new Passes.Mutators.Substitutor(e =>
        {
            if (e is Call c && c.Target is Fusion f)
            {
                // CompilerServices.Rewrite(f.Body, new[] { new FoldRepeatMarker() }, new());
                var effectVars = f.Name.Split("_").Chunk(2).SelectMany(list =>
                {
                    var originName = string.Join("_", list);
                    return set[originName];
                }).ToHashSet().ToArray();
                return c.With(target: BucketFusion.FromNormalFusion(f, effectVars));
            }

            return null;
        });
        mutator.Visit(post, Unit.Default);
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
}

internal sealed class BucketFusionGroupMutator : Passes.Mutators.FusionGroupMutator
{
    public BucketFusionGroupMutator(IMergeRewriteRule preOrderfusionRule, RunPassContext passOptions)
        : base(preOrderfusionRule, passOptions)
    {
    }

    public override bool MergedFusionCheckCallBack(Fusion mergedFusion, HashSet<Fusion> candidateFusions)
    {
        Console.WriteLine("-----------------");
        Console.WriteLine(mergedFusion.Name);
        Console.WriteLine("-----------------");
        foreach (var candidateFusion in candidateFusions)
        {
            Console.WriteLine(candidateFusion.Name);
        }

        Console.WriteLine("-----------------");

        // 回避反卷积，反卷积的shape表达式目前会引起重复的计算
        if (mergedFusion.Name.Contains("Conv2DTranspose", StringComparison.Ordinal) ||
            candidateFusions.Any(f => f.Name.Contains("Conv2DTranspose", StringComparison.Ordinal)))
        {
            return false;
        }

        return true;
    }
}
