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
using Nncase.PatternMatch;
using static Nncase.PatternMatch.Utility;
using static Nncase.Utilities.ReplaceUtility;
using static Nncase.Passes.Rules.ShapeBucket.ShapeBucketHelper;
using Tuple = Nncase.IR.Tuple;

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

[RuleGenerator]
public partial class MergeTupleFusion : RewriteRule<Pattern>
{
    public override Pattern Pattern => IsTuple("tuple", new VArgsPattern(
        list =>
        {
            return Enumerable.Range(0, list.Length).Select(_ =>
                IsWildcard(null, field => field is Call { Target: BucketFusion } && field.Users.Count == 1)).ToArray();
        }, null));

    // todo: merge these
    Expr? GetReplace(Tuple tuple)
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
        var newBody = MergeBucketFusion.ReplaceClone(new IR.Tuple(fieldBodys), oldParamsToNewArg.ToArray());
        var newFusion = new BucketFusion("stackvm", newBody, newParams.ToArray(), new Var[] { });
        var newCall = new Call(newFusion, newArgs.ToArray());
        return newCall;
    }
}

public class MergeBucketFusion : ModulePass
{
    private static int counter = 0;

    private static string relPath => counter.ToString();

    protected override Task<IRModule> RunCoreAsync(IRModule input, RunPassContext context)
    {
        // 1. save effect var info
        var main = (Function)input.Entry!;

        // var post = MergePrevFusion(main, set);
        // MergeMultiUsers(post);
        // return Task.FromResult(input);

        var hashcode = main.GetHashCode();
        int loop = 0;
        while (loop < 20)
        {
            var mergePrevPost = MergePrevFusion(main);
            MergeMultiUsers(mergePrevPost);
            MergeTupleFusion(mergePrevPost);
            var post = MergeMultiUsersSingleCall(mergePrevPost);
            var postHashCode = post.GetHashCode();
            if (hashcode != postHashCode)
            {
                counter++;
            }
            else
            {
                break;
            }

            hashcode = postHashCode;
            loop++;
            // todo: multi merge
        }
        return Task.FromResult(input);
    }

    private static void MergeTupleFusion(Function mergePrevPost) => CompilerServices.Rewrite(mergePrevPost, new[] { new MergeTupleFusion() }, new());

    private Expr MergeMultiUsersSingleCall(Expr body)
    {
        return CompilerServices.Rewrite(body, new[] { new MultiUserCallToFusion() }, new());
    }

    private Function MergePrevFusion(Function main)
    {
        // 1. get origin info
        var s = new SearchBucketFusion();
        s.Visit(main);
        var set = s.FusionEffectVars();

        // 2. merge
        var post = MergeFusion(main);
        DumpIR(post, "AfterMergeFusion", relPath);

        // 3. translate fusion to BucketFusion
        TranslateFusionToBucket(set, post, CompileSession);
        DumpIR(post, "AfterTranslateFusion", relPath);
        return post;
    }

    private static void MergeMultiUsers(Function post)
    {
        IRHelpers.DCE(post);
        DumpIR(post, "AfterDCE", relPath);
        var c = new ReplaceVisitor();
        c.Replace(post);
        DumpIR(post, "AfterMergeUser", relPath);
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
            // var fusionPattern = new FusionBucket().Pattern;
            if (expr is Call outerCall && outerCall.Target is BucketFusion fusion)
            {
                // var outerCall = (Call)matchResult["outerCall"];
                // var fusion = (BucketFusion)matchResult["fusion"];

                Console.WriteLine($"Match {fusion.Name}");
                // DumpIR(outerCall, $"{counter}_MergeBefore", relPath);
                // for (var i = 0; i < outerCall.Users.Count; i++)
                // {
                    // DumpIR(outerCall.Users.ToArray()[i], $"{counter}_User_{i}_MergeBefore", relPath);
                // }
                // DumpIR(_root, "OriginRoot", relPath);

                var newCall = MergeMultiUserFusion(outerCall, fusion);
                if (newCall != null)
                {
                    // DumpIR(outerCall, $"{counter}_OriginAfter", relPath);
                    // DumpIR(newCall, $"{counter}_MergeAfter", relPath);
                    // DumpIR(_root, "rootBeforeMerge", relPath);
                    var users = outerCall.Users.ToArray();
                    for (var i = 0; i < users.Length; i++)
                    {
                        // todo:这里计算的不对，不一定是按照顺序引用的，也可能某一个引用了多次，但是构造的时候是按照
                        // 原始body, users的输出开始构造的
                        // 第几个user的use，
                        var newOperand = newCall.CheckedType is TupleType ? newCall[i] : newCall;
                        // DumpIR(newOperand, $"newOperand_{i}", relPath);
                        ReplaceAllUsesWith(users[i], newOperand);
                    }

                    Console.WriteLine();
                    // ReplaceAllUsesWith(outerCall, newCall);
                    // DumpIR(_root, "rootAfterMerge", relPath);
                    MergeBucketFusion.counter++;
                    return newCall;
                }
            }

            return expr;
        }

        protected override Expr DefaultVisitLeaf(Expr expr) => expr;
    }

    private static bool detectedRing(Call outerCall)
    {
        var users = outerCall.Users.ToArray();
        var userArgs = users.SelectMany(user => ((Call)user).Arguments.ToArray()).ToArray();
        foreach (var arg in userArgs)
        {
            // todo: 必须检查是否到了limit,到了就错了，同时也不能继续visit下去了
            var list = new FindExpr().Run(arg, users, outerCall, expr =>
            {
                if (expr is Const)
                {
                    return false;
                }
                // todo:这里的检查好像就没意义了
                return users.Contains(expr);
            });
            if (list.Count > 0)
            {
                return true;
            }
        }

        return false;
    }

    private static Expr? MergeMultiUserFusion(Call outerCall, BucketFusion fusion)
    {
        var users = outerCall.Users.ToArray();
        // todo: not support
        if (users.Count(user => user is Tuple) != 0)
        {
            Console.WriteLine("HasTuple");
            return null;
        }

        // todo:如果不是所有的都是valid的，那么能合并吗？？
        var userInfos = CollectUsers(outerCall);

        // todo: support only one user, because merge fusion rule is not enough
        // maybe a error
        // if (userInfos.Length < 2)
        // {
        //     return null;
        // }

        // has invalid
        if (userInfos.Length != outerCall.Users.Count)
        {
            Console.WriteLine("not all call");
            return null;
        }

        if (detectedRing(outerCall))
        {
            Console.WriteLine("HasRing");
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
        var argMap = MakeNewVarsMap(userInfos, fusionDict, outerCall);
        // for (int i = 0; i < argMap.Length; i++)
        // {
        //     Console.WriteLine(argMap[i].Item1.GetHashCode());
        //     // Console.WriteLine(((Call)argMap[i].Item1).Target);
        //     Console.WriteLine(argMap[i].Item2.Name);
        // }
        var newUsers = MakeNewUserExpr(userInfos, outerCall, argMap);
        var newBody = MakeNewBody(newUsers);
        // DumpIR(newBody, "newBody", relPath);

        // todo: args去除重复，在不需要更新的情况下不进行更新
        var newArgs = argMap.NewArgs();
        var newParams = argMap.NewParams;

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
    record FusionVarMapper(Var[] NewParams, (Expr UserArg, Var RelativeNewVar, Var OldVar)[] ArgMap)
    {
        public Dictionary<Var, Var> oldToNewParam()
        {
            return ArgMap.ToDictionary(info => info.OldVar, info => info.RelativeNewVar);
        }

        public Expr[] NewArgs()
        {
            return ArgMap
                .Where(pair => NewParams.Contains(pair.RelativeNewVar))
                .Take(NewParams.Length)
                .Select(pair => pair.UserArg)
                .ToArray();
        }
    }

    private static FusionVarMapper MakeNewVarsMap(UserInfo[] userInfos, (Expr, Var)[] fusionDict, Call outerCall)
    {
        var originArgs = fusionDict.Concat(userInfos.SelectMany(info =>
        {
            var user = info.User;
            if (user.Target is BucketFusion fusion)
            {
                return user.Arguments.ToArray().OfNoConst().Zip(fusion.Parameters.ToArray());
            }

            throw new NotImplementedException();
        })).ToArray();

        var users = userInfos.Select(x => x.User).ToArray();
        // 保证var顺序以及字典
        // 初始param应该包含fusion的
        var fusionParams = fusionDict.Select(pair => pair.Item2).ToArray();
        // arg到var的映射，并不需要使用fusion的信息设置为初始值，因为合并的是user
        var result = originArgs.Aggregate(
            (new (Expr UserArg, Var RelativeNewVar, Var OldVar)[] { }, fusionParams),
            (sum, pair) =>
            {
                var (totalDict, totalVars) = sum;
                var (arg, oldVar) = pair;
                var fusionResult = FindFirst(fusionDict, arg);
                // todo: 这里没有判断成功
                if (fusionResult != null)
                {
                    return (totalDict.Append((arg, fusionResult!, oldVar)), totalVars);
                }

                var result = FindFirst(totalDict.Select(tuple => (tuple.UserArg, tuple.RelativeNewVar)).ToArray(), arg);
                if (result != null)
                {
                    return (totalDict.Append((arg, result!, oldVar)), totalVars);
                }

                // todo: maybe put fusion body in this is better?
                // arg is outerCall or arg is otherOuterCall
                if (arg == outerCall)
                {
                    return (totalDict, totalVars);
                }

                // 其他参数被合并进来了，也就不需要再创建新的var,但是后面替换的时候也要对这种情况进行替换
                if (users.Contains(arg))
                {
                    return (totalDict, totalVars);
                }

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
    internal static Expr ReplaceClone(Expr originBody, params (Var, Expr)[] originVarAndExpr)
    {
        var call = originBody.Clone();
        var finder = new FindVar();
        finder.Visit(call);
        var newVars = finder.Vars;
        originVarAndExpr.ForEach(pair =>
        {
            var (v, newExpr) = pair;
            var varShouldBeReplaced = newVars.FindFirst(newVar => newVar.Name == v.Name);
            if (varShouldBeReplaced == null)
            {
                throw new InvalidOperationException();
            }
            ReplaceExpr(call, varShouldBeReplaced, newExpr);
        });
        return call;
    }

    private static Expr[] MakeNewUserExpr(UserInfo[] userInfos, Call outerCall, FusionVarMapper argMap)
    {
        var users = userInfos.Select(x => x.User).ToArray();
        var usersName = users.Select(user => user.Target).OfType<BucketFusion>().Select(x => x.Name).ToArray();
        // 不能加入原始的body，因为所有的输出都被合并了，只有合并进来的输出了
        var originBody = new IR.Tuple(users.ToArray());
        // var newOriginBody =
            // new IR.Tuple(new Expr[] { body }
                // .Concat(users.Select(user => user.Target).OfType<BucketFusion>().Select(f => f.Body)).ToArray());
        var newOriginBody = originBody.Clone();
        var finder = new FindVar();
        finder.Visit(newOriginBody);
        var newVars = finder.Vars;
        // todo: this is unused
        var map = argMap.oldToNewParam();
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
                        return ReplaceExpr(sum, pair.Item1, pair.Item2);
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

    // private static Expr[] MakeNewUserExprs(UserInfo[] userInfos, Expr body, FusionVarMapper argMap)
    // {
    //     // user的call的args中有两部分var,
    //     // 1：用到fusion的部分，这部分需要替换为fusion的body。而这一步需要先将fusion的body用一个var代替，之后再用body替换掉这个var
    //     // 第一部分 不能直接替换为fusion的body，因为如果两个user互相有子表达式关系的话会导致重复的call
    //     // 2：替换完以后需要将整个body中所有引用的其他var替换为整个fusion的var
    //     var users = userInfos.Select(x => x.User).ToArray();
    //     var fusionMap = users.Select(user => user.Target).OfType<BucketFusion>().ToDictionary(fusion => fusion, fusion => new Var());
    //     // make user expr, replace call with fusion body
    //     // DumpIR(new IR.Tuple(users), "BeforeMakeNewUser", relPath);
    //
    //     // 首先不可能出现互相引用，那么可以在出现引用的时候先修改另一个，然后将修改的而结果拿来使用
    //     // FusionIndexInUserArg的不可能被替换
    //
    //     // var fusionMapInfo = fusionMap.Select(pair => ((Var)pair.Value, (Expr)pair.Key.Body)).ToArray();
    //     // 第一部分：所有fusion的var所关联的其他fusion的信息，用于之后统一替换
    //     var fusionReplaceInfo = MakeFusionReplaceInfo(userInfos, body, users, fusionMap)
    //         // .Concat(fusionMapInfo)
    //         .Concat(
    //             argMap.oldToNewParam()
    //                 .Select(pair => (pair.Key, (Expr)pair.Value))
    //                 .ToArray())
    //         .ToArray();
    //
    //     // todo: output顺序问题
    //     // 直接构建新的body，可以解决互相为子表达式的问题
    //     var newOriginBody =
    //         new IR.Tuple(new Expr[] { body }
    //             .Concat(users.Select(user => user.Target).OfType<BucketFusion>().Select(f => f.Body)).ToArray());
    //     // 根据arg map替换
    //     // 1. arg to fusion arg
    //
    //     // 2. arg to fusion body
    //     // 3. old to new
    //     DumpIR(newOriginBody, "newOriginBody", relPath);
    //     // clone
    //     var call = newOriginBody.Clone();
    //     var finder = new FindVar();
    //     finder.Visit(call);
    //     var newVars = finder.Vars;
    //     // replace
    //     var resFusionMap = fusionMap.ToDictionary(pair => pair.Value, pair => pair.Key);
    //     var mutator = new Passes.Mutators.Substitutor(e =>
    //     {
    //         if (fusionMap.TryGetValue(e, out var result))
    //         {
    //
    //         }
    //     });
    //     // 然后替换每个fusion中的arg为对应的var
    //     DumpIR(newUsers, "newUsers", relPath);
    //     return ((IR.Tuple)newUsers).Fields.ToArray();
    //
    // }

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
                    // arg是多个user的call,替换对应的body
                    if (i == userInfo.FusionIndexInUserArg)
                    {
                        return (param, body);
                    }

                    // todo: arg是其他的fusion的话替换为其fusion的body，但是其中的var可能要后面才能替换了
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

    private static UserInfo[] CollectUsers(Expr outerCall)
    {
        var users = outerCall.Users.ToArray();
        // todo: process for getItem multi user
        // if (outerCall.CheckedType is TupleType)
        // {
        //     users = users.Select(user =>
        //     {
        //         return ((Call)user).Arguments[IR.Tensors.GetItem.Input.Index];
        //     }).ToArray();
        // }

        var outputs = outerCall.Users
            .Select((user, userIndex) =>
            {
                // 找到fusion在user的arguments的哪个index
                if (user is Call userCall)
                {
                    var valid = userCall.Target switch
                    {
                        // todo:目前只支持全是fusion的情况
                        Op op => false,
                        // todo: check effect var
                        BucketFusion => true,
                        // what this
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
                // todo: 需要避免环的出现，users里面的input有可能依赖于其他output的结果
                else
                {
                    var valid = false;
                    return (user, userIndex, valid);
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

    private static void TranslateFusionToBucket(Dictionary<string, Var[]> set, Function post, CompileSession seesion)
    {
        var inputDimsVars = InputDimVars(seesion);
        var mutator = new Passes.Mutators.Substitutor(e =>
        {
            if (e is Call c && c.Target is Fusion f)
            {
                // CompilerServices.Rewrite(f.Body, new[] { new FoldRepeatMarker() }, new());
                var effectVars = new Var[] { };
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
