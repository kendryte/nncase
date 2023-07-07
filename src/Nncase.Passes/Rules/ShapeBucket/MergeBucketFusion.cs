using System;
using System.Collections.Generic;
using System.Linq;
using System.Reactive;
using System.Threading.Tasks;
using DryIoc.ImTools;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Toolkit.HighPerformance;
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

        // 3. translate fusion to BucketFusion
        TranslateFusionToBucket(set, post);
        return Task.FromResult(input);
    }

    record UserInfo(Call User, int UserIndex, int FusionIndexInUserArg);

    internal class CallVisitor : ExprVisitor<Expr, Unit>
    {
        protected override Expr VisitLeafCall(Call expr)
        {
            var fusionPattern = new FusionBucket().Pattern;
            if (CompilerServices.TryMatchRoot(expr, fusionPattern, out var matchResult) && expr.Users.Count > 1)
            {
                var outerCall = (Call)matchResult["outerCall"];
                var fusion = (BucketFusion)matchResult["fusion"];
                MergeMultiUserFusion(outerCall, fusion);
            }

            return expr;
        }
    }

    private static void MergeMultiUserFusion(Call outerCall, BucketFusion fusion)
    {
        var userInfos = CollectUsers(outerCall);

        // maybe a error
        if (userInfos.Length < 2)
        {
            return;
        }

        // todo: with tuple
        // 1. all user are fusion
        // 2. some user are fusion
        // 3. no fusion
        var newUsers = MakeNewUserExpr(userInfos, fusion.Body);
        var newBody = MakeNewBody(newUsers);
        var newParams = MakeNewParams(fusion, userInfos);
        var newArgs = MakeNewArgs(outerCall, userInfos);
        var newFusion = MakeNewFusion(newBody, fusion, newParams);
        var newCall = MakeNewCall(newFusion, newArgs);
        UpdateUserOfUserExpr(newCall, userInfos);
    }

    // update user of user expr, replace call with getItem(call)
    private static void UpdateUserOfUserExpr(Call newCall, UserInfo[] userInfos)
    {
        userInfos.ForEach(userInfo =>
        {
            userInfo.User.Users.Select(userOfUser =>
            {
                if (userOfUser is Call call)
                {
                    // todo: 会不会有其他情况
                    return ReplaceExpr(call, userInfo.User, newCall[userInfo.UserIndex]);
                }

                throw new NotImplementedException();
            });
        });
    }

    private static BucketFusion MakeNewFusion(Expr body, BucketFusion fusion, Var[] newParams)
    {
        // todo: EffectVar
        return new BucketFusion("k230", body, newParams, fusion.EffectVar);
    }

    private static Call MakeNewCall(BucketFusion fusion, Expr[] args)
    {
        return new Call(fusion, args);
    }

    private static Expr[] MakeNewArgs(Call outerCall, UserInfo[] userInfos)
    {
        var fusionArgs = outerCall.Arguments.ToArray();
        var newParams = userInfos.SelectMany(userInfo =>
        {
            var user = userInfo.User;
            return user.Arguments.ToArray().OfNoConst().ToArray().RemoveAt(userInfo.FusionIndexInUserArg);
        }).ToArray();
        return fusionArgs.Concat(newParams).ToArray();
    }

    private static Var[] MakeNewParams(BucketFusion fusion, UserInfo[] userInfos)
    {
        var fusionParams = fusion.Parameters.ToArray();
        var newParams = userInfos.SelectMany(userInfo =>
        {
            // todo: process for Fusion
            var usersArgs = userInfo.User.Arguments.ToArray().OfNoConst().ToArray();
            return usersArgs.Select(arg => new Var(arg.CheckedType)).ToArray();
        }).ToArray();
        return fusionParams.Concat(newParams).ToArray();
    }

    private static Expr MakeNewBody(Call[] newUsers)
    {
        return new IR.Tuple(newUsers);
    }

    private static Call[] MakeNewUserExpr(UserInfo[] userInfos, Expr body)
    {
        // todo: fusion deconstruct
        // make user expr, replace call with fusion body
        return userInfos.Select(userInfo => ReplaceCallParams(userInfo.User, (userInfo.FusionIndexInUserArg, body)))
            .ToArray();
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
                var argInUserArg = user.Arguments.IndexOf(outerCall);
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
        DumpIR(post, "AfterTranslateFusion");
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

        DumpIR(post, "AfterMergeFusion");
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
