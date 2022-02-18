using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Microsoft.Extensions.Hosting;
using Nncase.IR;
using Nncase.Pattern;
using Nncase.Pattern.Math;
using Nncase.Transform;
using Xunit;
using static Nncase.IR.F.Math;
using static Nncase.IR.F.Tensors;
using static Nncase.IR.TypePatternUtility;
using static Nncase.Pattern.F.Math;
using static Nncase.Pattern.F.Tensors;
using static Nncase.Pattern.Utility;

namespace Nncase.Tests.MatchTest;

public class EGraphMatchTestFactory : ReWriteTest.RewriteFixtrue
{
    public EGraphMatchTestFactory(IHost host) : base(host) { }

    public static IEnumerable<object[]> Data =>
      new List<object[]>
      {
             new object[] {"FoldNopTransposeCase2", new ReWriteTest.FoldNopTransposeCase2().PreExpr, new Transform.Rule.FoldTranspose(), new[]{ 1} },
             new object[] {"FoldNopTransposeCase1", new ReWriteTest.FoldNopTransposeCase1().PreExpr, new Transform.Rule.FoldTranspose(), new[]{ 1} },
      };

    [Theory]
    [MemberData(nameof(DataOne))]
    public void RunOne(string Name, Expr Pre, IRewriteRule Rule, int[] targets) => RunCore(Name, Pre, Rule, targets);

    public static IEnumerable<object[]> DataOne => Data.Take(1);
    public static IEnumerable<object[]> DataAll => Data.Skip(1);

    protected void RunCore(string Name, Expr Pre, IRewriteRule Rule, int[] targets)
    {
        passOptions.SetName($"EGraphMatchTest/{Name}");
        Assert.True(Pre.InferenceType());
        var eGraph = new EGraph();
        eGraph.Add(Pre, out var root);
        EGraphPrinter.DumpEgraphAsDot(eGraph, Path.Combine(passOptions.PassDumpDir, $"pre"));
        Pre.DumpExprAsIL("pre", passOptions.PassDumpDir);
        foreach (var (pat, target) in Rule.Patterns.Zip(targets))
        {
            var results = EGraphMatcher.Match(eGraph, pat);
            Assert.Equal(target, results.Count);
            if (passOptions.DumpLevel > 1)
                EGraphPrinter.DumpEgraphAsDot(eGraph, results,
                 Path.Combine(passOptions.PassDumpDir, $"V{eGraph.Version}"));
        }
    }

    [Theory]
    [MemberData(nameof(DataAll))]
    public void RunAll(string Name, Expr Pre, IRewriteRule Rule, int[] targets) => RunCore(Name, Pre, Rule, targets);
}

public class UnitTestEGraphMatch
{

    [Fact]
    public void TestWildCardRecursion()
    {
        WildcardPattern wcx = "x", wcy = "y";
        var pat = wcx + (wcy + IsConst(1));
        Var x = "x", y = "y";
        Expr expr = x + (y + 1);

        var res = EGraphMatcher.Match(expr, pat);
        Assert.Single(res);
        Assert.IsType<Var>(res[0][wcx]);
        Assert.IsType<Var>(res[0][wcy]);
    }

    [Fact]
    public void TestWildCardRecursion2()
    {
        WildcardPattern wcx = "x", wcy = "y";
        var pat = wcx + (wcy + IsConst(1));
        Var x = "x", y = "y";
        Expr expr = x + (y + 1);
        var res = EGraphMatcher.Match(expr, pat);
        Assert.Single(res);
    }

    [Fact]
    public void TestMatchOpAdd()
    {
        var wc1 = IsWildcard();
        var pat = wc1 + 1;

        var a = new Var("a");
        var wce1 = a * 100 - 32 / 320;
        var e = wce1 + 1;
        var g = new EGraph();
        g.Add(e);

        var matchs = EGraphMatcher.Match(g, pat);
        Assert.Single(matchs);
        var result = matchs[0];
        Assert.Equal(result[wc1], wce1);
    }

    [Fact]
    public void TestMatchOpOR()
    {
        var x = new Var("a");
        var y = x + 10;
        var y1 = y - 10;

        var px = IsWildcard();
        var py = IsBinary(op => op is (BinaryOp.Add or BinaryOp.Sub), px, 10);

        var matchs = EGraphMatcher.Match(y, py);
        Assert.Single(matchs);

        var matchs2 = EGraphMatcher.Match(y1, py);
        Assert.Equal(2, matchs2.Count);

        var py1 = IsUnary(UnaryOp.Abs, px);
        Assert.Empty(EGraphMatcher.Match(y1, py1));
    }

    [Fact]
    public void MatchFunction()
    {
        Var x = "x";
        Var y = "y";

        WildcardPattern wc1 = "x";
        WildcardPattern wc2 = "y";

        Expr func = new Function(x + y - 1200, x, y);
        ExprPattern pat_1 = new FunctionPattern(x + y - 1200, wc1, wc2);

        ExprPattern pat_2 = new FunctionPattern(x - y, wc1, wc2);

        var res_1 = EGraphMatcher.Match(func, pat_1);
        Assert.Single(res_1);

        var res_2 = EGraphMatcher.Match(func, pat_2);
        Assert.Empty(res_2);
    }

    [Fact]
    public void TestMatchVArgs()
    {

        WildcardPattern wc = "x";
        List<ExprPattern> wcs = new();

        var nest_tuple = new IR.Tuple(4, 5, 6);
        var tuple = new IR.Tuple(1, nest_tuple, 3);
        Expr expr = Concat(tuple, 0);


        ExprPattern vpat = Concat(IsTuple(IsVArgsRepeat((n, param) =>
        {
            for (int i = 0; i < n; i++)
            {
                var wc = IsWildcard();
                param.Add(wc);
                wcs.Add(wc);
            }
        },
        (match, param) =>
        {
            if (!match)
            {
                param.Clear();
                wcs.Clear();
            }
        }
        )), 0);


        var eMatches = EGraphMatcher.Match(expr, vpat);
        Assert.Single(eMatches);
        var eMatch = eMatches[0];
        Assert.Equal(eMatch[wcs[0]], tuple[0]);
        Assert.Equal(eMatch[wcs[1]], tuple[1]);
        Assert.Equal(eMatch[wcs[2]], tuple[2]);
    }

    [Fact]
    public void TestMatchVArgsTwice()
    {

        ConstPattern wcaxis = IsConst();
        List<ConstPattern> wccons = new();

        var tuple_lhs = new IR.Tuple(1, new Var(), 3);
        var tuple_rhs = new IR.Tuple(4, 5, 6);
        Expr expr = Concat(tuple_lhs, 0) + Concat(tuple_rhs, 1);

        ExprPattern vpat = Concat(IsTuple(IsVArgsRepeat((n, param) =>
        {
            for (int i = 0; i < n; i++)
            {
                var wc = IsConst();
                param.Add(wc);
                wccons.Add(wc);
            }
        },
        (match, param) =>
        {
            if (!match)
            {
                param.Clear();
                wccons.Clear();
            }
        }
        )), wcaxis);


        var eMatches = EGraphMatcher.Match(expr, vpat);
        Assert.Single(eMatches);
        var eMatch = eMatches[0];
        Assert.Equal(eMatch[wccons[0]], tuple_rhs[0]);
        Assert.Equal(eMatch[wccons[1]], tuple_rhs[1]);
        Assert.Equal(eMatch[wccons[2]], tuple_rhs[2]);
    }

    [Fact]
    public void TestMatchVArgsRecursion()
    {

        Var x = "x";
        Const y = 4;
        Expr z = (Const)1 + 2;

        Const perm = 123;
        Expr expr = Concat(new IR.Tuple(
          Transpose(x, perm),
          Transpose(y, perm),
          Transpose(z, perm)), 0);

        WildcardPattern wc = "wc", wcperm = "perm", wcaxis = "axis";
        List<WildcardPattern> wcin = new();
        var wcvargs = IsVArgsRepeat((n, param) =>
        {
            for (int i = 0; i < n; i++)
            {
                var input = IsWildcard();
                param.Add(Transpose(input, wcperm));
                wcin.Add(input);
            }
        });
        var pattern = Concat(IsTuple(wcvargs), wcaxis);


        var results = EGraphMatcher.Match(expr, pattern);
        Assert.Single(results);
        var result = results[0];
        Assert.Equal(result[wcin[0]], x);
        Assert.Equal(result[wcin[1]], y);
        Assert.Equal(result[wcin[2]], z);
        Assert.Equal(result[wcperm], perm);
        Assert.Equal(result[wcaxis], (Const)0);
    }

    [Fact]
    public void TestMatchSameConstPatternTwice()
    {
        var x = (Const)1;
        Expr expr = (x * x) + 12 - x;
        var xpat = IsConst();
        var result = EGraphMatcher.Match(expr, IsBinary(xpat, xpat));
        Assert.Single(result);

        var result2 = EGraphMatcher.Match((x * 2) + 12 - x, IsBinary(xpat, xpat));
        Assert.Empty(result2);
    }

    [Fact]
    public void TestMatchSameWildCardPatternTwice()
    {
        var x = (Const)1;
        Expr expr = (x * x) + 12 - x;
        var xpat = IsWildcard();
        var result = EGraphMatcher.Match(expr, IsBinary(xpat, xpat));
        Assert.Single(result);

        var result2 = EGraphMatcher.Match((x * 2) + 12 - x, IsBinary(xpat, xpat));
        Assert.Empty(result2);
    }
}

