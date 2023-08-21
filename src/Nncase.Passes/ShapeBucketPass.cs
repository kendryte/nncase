// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Reactive;
using System.Threading.Tasks;
using DryIoc.ImTools;
using Nncase.IR;
using Nncase.Passes.Rules.Lower;
using Nncase.Passes.Rules.Neutral;
using Nncase.Passes.Rules.ShapeBucket;
using Nncase.Passes.Rules.ShapeExpr;
using BaseFunction = Nncase.IR.BaseFunction;

namespace Nncase.Passes;

public class ShapeBucketPass : FunctionPass
{
    private readonly List<IRewriteRule> _toFusion = new();

    public ShapeBucketPass()
    {
        var singleVar = CompileSession.CompileOptions.ShapeBucketOptions.VarMap.Values.SelectMany(x => x).OfType<Var>().ToHashSet().Count <= 1;

        _toFusion.Add(new MatmulToFusion());
        _toFusion.Add(new Conv2DToFusion());
        _toFusion.Add(new TFConv2DTransposeToFusion());
        _toFusion.Add(new Conv2DTransposeToFusion());

        _mergeCall.Add(new MergePrevCallToFusion());
        _mergeCall.Add(new MergePrevMarkerToFusion());
        _mergeCall.Add(new MergeNextCallToFusion());
        _mergeCall.Add(new MergeNextMarkerToFusion());

        _lostToFusion.Add(new TransposeToFusion());
        _lostToFusion.Add(new UnaryToFusion());
        _lostToFusion.Add(new ActToFusion());
        if (singleVar)
        {
            _lostToFusion.Add(new BinaryToFusion());
        }

        _clearMarker.Add(new ClearFusionOuterMarker());
        _clearMarker.Add(new RemoveMarker());

        _mergeRule.Add(new MultiUserCallToFusion());
        _mergeRule.Add(new MergeTupleFusion());
        _mergePass.Add(new MergeSeqBucketFusion());
        _mergePass.Add(new MergeMultiUsersFusion());

        _simplify.Add(new FoldStackGetItem());
        _simplify.Add(new FoldConstCall());
        _simplify.Add(new FoldShapeOf());
        _simplify.Add(new FoldTwoReshapes());
        _simplify.Add(new FoldTwoCasts());
        _simplify.Add(new FoldTwoSlices());
        _simplify.Add(new FoldNopBinary());
        _simplify.Add(new FoldNopCast());
        _simplify.Add(new FoldNopReshape());
        _simplify.Add(new FoldNopSlice());
        _simplify.Add(new FoldIf());
    }

    private readonly List<IRewriteRule> _mergeCall = new();
    private readonly List<IRewriteRule> _lostToFusion = new();
    private readonly List<IRewriteRule> _clearMarker = new();
    private readonly List<IRewriteRule> _bucket = new();
    private readonly List<IRewriteRule> _simplify = new();
    private readonly List<IRewriteRule> _mergeRule = new();
    private readonly List<FunctionPass> _mergePass = new();

    private RunPassContext _context = new();

    private Function _main;

    private bool debugMode = true;

    protected override Task<BaseFunction> RunCoreAsync(BaseFunction input, RunPassContext context)
    {
        var t = new SimpleTimer("ShapeBucketPass");
        if (input is not Function)
        {
            return Task.FromResult(input);
        }

        _context = context;
        _main = (Function)input;

        GreedyMerge();
        ShapeBucket(input, context);
        Rebuild();
        Rewrite(_simplify);
        return Task.FromResult((BaseFunction)_main);
    }

    private void ShapeBucket(BaseFunction input, RunPassContext context)
    {
        var shapeList = new Dictionary<BucketFusion, FusionShapeData[]>();
        Task.FromResult(new RecordFusionShape(shapeList).RunAsync(input, context));
        _bucket.Add(new FusionBucket(shapeList));
        Rewrite(_bucket);
    }

    private void MergeFusion(bool greedy)
    {
        // 修好shape表达式的复制之前不能用
        if (!greedy)
        {
            return;
        }

        if (debugMode)
        {
            return;
        }

        Rewrite(_mergeRule);
        while (true)
        {
            var preHash = _main.GetHashCode();
            Rewrite(_mergeRule);
            foreach (var pass in _mergePass)
            {
                RunPass(pass);
            }

            var postHash = _main.GetHashCode();
            if (preHash != postHash)
            {
                break;
            }
        }
    }

    private void Check()
    {
        FusionChecker.CheckErrorVar(_main, _main.Parameters.ToArray());
        FusionChecker.CheckRepeat(_main);
    }

    private Function RunPass(FunctionPass pass)
    {
        var result = (Function)pass.RunAsync(_main, _context).Result;
        Check();
        return result;
    }

    private void GreedyMerge()
    {
        Rewrite(_toFusion);
        Rewrite(_mergeCall);
        Rewrite(_lostToFusion);
        Rewrite(_mergeCall);
        Rewrite(_clearMarker);
        MergeFusion(true);
    }

    private void Rebuild()
    {
        if (debugMode)
        {
            return;
        }

        // todo: dynamic only
        Rewrite(_toFusion);
        Rewrite(_mergeCall);
        Rewrite(_clearMarker);
        MergeFusion(false);
        ShapeBucket(_main, _context);
    }

    private Function Rewrite(List<IRewriteRule> rules)
    {
        OnPassStartAsync(_main, _context);
        var result = (Function)CompilerServices.Rewrite(_main, rules, _context);
        Check();
        OnPassEndAsync(_main, _context);
        return result;
    }
}

public static class FusionChecker
{
    public static void CheckRepeat(Expr call)
    {
        // todo: 检查所有fusion里面的param有没有重复名字的
        // todo: 检查有没有fusion名字重复的
        var c = new CheckFusionCallVisitor();
        c.Visit(call);
        c.Check();
    }

    public static void CheckErrorVar(Expr body, Var[] vars)
    {
        var f = new FindVar();
        f.Visit(body);
        if (!f.Vars.All(vars.Contains))
        {
            Console.WriteLine(string.Join(", ", f.Vars.Select(x => x.Name).ToArray()));
            throw new InvalidOperationException("Has Invalid Var In Body");
        }
    }
}

internal sealed class CheckFusionCallVisitor : ExprWalker
{
    private readonly HashSet<string> _callName = new();
    private readonly Dictionary<string, (string, BucketFusion)> _errorFusion = new();

    private readonly HashSet<string> _fusionName = new();
    private readonly HashSet<string> _repeatFusion = new();

    private readonly HashSet<string> _fusionParamsName = new();
    private readonly HashSet<string> _repeatParamFusion = new();

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
