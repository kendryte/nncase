using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using Nncase.IR;
using Nncase.Passes.Rules.Lower;
using Nncase.Passes.Rules.Neutral;
using Nncase.Passes.Rules.ShapeBucket;
using Nncase.Passes.Rules.ShapeExpr;
using BaseFunction = Nncase.IR.BaseFunction;

namespace Nncase.Passes;

public class ShapeBucketPass : FunctionPass
{
    public ShapeBucketPass()
    {
        var singleVar = CompileSession.CompileOptions.ShapeBucketOptions.VarMap.Values.SelectMany(x => x).OfType<Var>().ToHashSet().Count <= 1;

        ToFusion.Add(new MatmulToFusion());
        ToFusion.Add(new Conv2DToFusion());
        ToFusion.Add(new Conv2DTransposeToFusion());

        MergeCall.Add(new MergePrevCallToFusion());
        MergeCall.Add(new MergeNextCallToFusion());

        LostToFusion.Add(new TransposeToFusion());
        LostToFusion.Add(new UnaryToFusion());
        LostToFusion.Add(new ActToFusion());
        if (singleVar)
        {
            LostToFusion.Add(new BinaryToFusion());
        }

        ClearMarker.Add(new ClearFusionOuterMarker());
        ClearMarker.Add(new RemoveMarker());

        Simplify.Add(new FoldStackGetItem());
        Simplify.Add(new FoldConstCall());
        Simplify.Add(new FoldShapeOf());
        Simplify.Add(new FoldTwoReshapes());
        Simplify.Add(new FoldTwoCasts());
        Simplify.Add(new FoldTwoSlices());
        Simplify.Add(new FoldNopBinary());
        Simplify.Add(new FoldNopCast());
        Simplify.Add(new FoldNopReshape());
        Simplify.Add(new FoldNopSlice());
        Simplify.Add(new FoldIf());

        Bucket.Add(new FusionBucket());
        Rebuild = ToFusion;
    }

    private readonly List<IRewriteRule> ToFusion = new();
    private readonly List<IRewriteRule> MergeCall = new();
    private readonly List<IRewriteRule> LostToFusion = new();
    private readonly List<IRewriteRule> ClearMarker = new();
    private readonly List<IRewriteRule> Bucket = new();
    private readonly List<IRewriteRule> Rebuild = new();
    private readonly List<IRewriteRule> Simplify = new();

    private RunPassContext context = new();

    private Function main;

    private Task<BaseFunction> Rewrite(List<IRewriteRule> rules)
    {
        OnPassStartAsync(main, context);
        Task.FromResult((BaseFunction)CompilerServices.Rewrite(main, rules, context));
        // todo: do check
        OnPassEndAsync(main, context);
        return Task.FromResult((BaseFunction)main);
    }

    protected override Task<BaseFunction> RunCoreAsync(BaseFunction input, RunPassContext context)
    {
        if (input is not Function)
        {
            return Task.FromResult(input);
        }

        this.context = context;
        main = (Function)input;
        Rewrite(ToFusion);
        Rewrite(MergeCall);
        Rewrite(LostToFusion);
        Rewrite(MergeCall);
        Rewrite(ClearMarker);
        var merge = new MergeBucketFusion();

        Rewrite(Bucket);
        Rewrite(Rebuild);
        Rewrite(Bucket);
        Rewrite(Simplify);
        return Task.FromResult((BaseFunction)main);
    }
}
