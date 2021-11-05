using System;
using Xunit;
using Nncase.Transform.Pattern;
using Nncase.Transform;
using Nncase.IR;
using System.Collections.Generic;
using Nncase.Transform.Pattern.Math;
using static Nncase.IR.F.Math;
using static Nncase.IR.F.Tensors;
using static Nncase.Transform.Pattern.Utility;
using static Nncase.Transform.Pattern.F.Math;
using static Nncase.Transform.Pattern.F.Tensors;
using static Nncase.Transform.EGraphMatcher;

namespace Nncase.Tests
{
    public class UnitTestEGraphPattern
    {

        [Fact]
        public void TestVarPattern()
        {
            Var e = new Var("x", AnyType.Default);
            ExprPattern ep = e;
            Assert.IsType<VarPattern>(ep);
            Assert.True(ep.MatchLeaf(e));
        }

        [Fact]
        public void TestConstantPattern()
        {
            var con = (Const)(1.1f);
            ExprPattern cp1 = con;
            Assert.IsType<ConstPattern>(cp1);

            ExprPattern cp2 = IsConst((float x) => x > 1.2f);
            ExprPattern cp3 = IsConst((int x) => x > 1);
            var cp4 = (ConstPattern)1.1f;

            Assert.True(cp1.MatchLeaf(con));
            Assert.False(cp2.MatchLeaf(con));
            Assert.False(cp3.MatchLeaf(con));
            Assert.True(cp4.MatchLeaf(con));
        }

        [Fact]
        public void TestConstantPatternEqual()
        {
            ConstPattern cp1 = (ConstPattern)1;
            ConstPattern cp2 = (ConstPattern)1;
            Dictionary<ConstPattern, int> d = new();
            d.Add(cp1, 1);
            Assert.NotEqual(cp1, cp2);
            Assert.DoesNotContain(cp2, d.Keys);
            ConstPattern cp3 = IsConst();
            ConstPattern cp4 = IsConst();
            d.Add(cp3, 1);
            Assert.NotEqual(cp3, cp4);
            Assert.DoesNotContain(cp4, d.Keys);
        }

        [Fact]
        public void TestWildcardPattern()
        {
            var wc = IsWildCard();
            Assert.IsType<WildCardPattern>(wc);
        }

        [Fact]
        public void TestWildCardPatternHash()
        {
            var wc = IsWildCard();
            var wc2 = new WildCardPattern();
            var wc3 = new WildCardPattern(wc2.Name);
            var d = new Dictionary<WildCardPattern, int>();
            d.Add(wc, 1);
            d.Add(wc2, 2);
            d.Add(wc3, 2);
        }

        [Fact]
        public void TestCallPattern()
        {
            var e = (Const)1 + Exp(10);

            var wc1 = IsWildCard();
            var wc2 = IsWildCard();
            var c = wc1 + wc2;
            Assert.IsType<CallPattern>(c);
            Assert.IsType<BinaryPattern>(c.Target);
            Assert.IsType<WildCardPattern>(c.Parameters[0]);
            Assert.IsType<WildCardPattern>(c.Parameters[1]);

            var c2 = IsBinary(BinaryOp.Add, wc1, wc2);

            var c3 = IsBinary(x => x is (BinaryOp.Div or BinaryOp.Sub), wc1, wc2);

            Assert.True(c.Target.MatchLeaf(e.Target));
            Assert.True(c2.Target.MatchLeaf(e.Target));
            Assert.False(c3.Target.MatchLeaf(e.Target));
        }

        [Fact]
        public void TestFunctionPattern()
        {
            var wc1 = IsWildCard();
            var wc2 = IsWildCard();
            var c = wc1 + wc2;
            var fp = new FunctionPattern(c, wc1, wc2);
            Assert.IsType<FunctionPattern>(fp);
            Assert.IsType<WildCardPattern>(fp.Parameters[0]);
            Assert.IsType<WildCardPattern>(fp.Parameters[1]);
            Assert.IsType<CallPattern>(fp.Body);
            Assert.IsType<WildCardPattern>(((CallPattern)fp.Body).Parameters[0]);
            Assert.IsType<WildCardPattern>(((CallPattern)fp.Body).Parameters[1]);

            var fp2 = new FunctionPattern(c, IsVArgs(wc1, wc2));
            Assert.IsType<WildCardPattern>(fp.Parameters[0]);
            Assert.IsType<WildCardPattern>(fp.Parameters[1]);
        }

        [Fact]
        public void TestTuplePattern()
        {
            var wc1 = IsWildCard();
            var wc2 = IsWildCard();
            var t = IsTuple(wc1, wc2);
            Assert.IsType<TuplePattern>(t);
            Assert.IsType<WildCardPattern>(t.Fields[0]);
            Assert.IsType<WildCardPattern>(t.Fields[1]);

            var t2 = IsTuple(IsVArgs(wc1, wc2));
            Assert.IsType<TuplePattern>(t2);
            Assert.IsType<WildCardPattern>(t2.Fields[0]);
            Assert.IsType<WildCardPattern>(t2.Fields[1]);
        }

        [Fact]
        public void TestVArgsPattern()
        {
            var wc = IsWildCard();
            var vwcs = new List<ExprPattern>();
            var pattern = IsVArgsRepeat((n, param) =>
            {
                for (int i = 0; i < n; i++)
                {
                    var wc = IsWildCard();
                    param.Add(wc);
                    vwcs.Add(wc);
                }
            },
            (match, param) =>
            {
                if (match == false)
                {
                    param.Clear();
                    vwcs.Clear();
                }
            }
            );

            var tuple = new IR.Tuple(1, new IR.Tuple(6, 7, 8), 3, 4);
            Assert.True(pattern.MatchLeaf(tuple.Fields));
            Assert.True(pattern[0].MatchLeaf(tuple[0]));
            Assert.True(pattern[1].MatchLeaf(tuple[1]));
            Assert.True(pattern[2].MatchLeaf(tuple[2]));
            Assert.True(pattern[3].MatchLeaf(tuple[3]));
        }

        [Fact]
        public void TestAltPattern()
        {
            var lhs = IsWildCard();
            var rhs = IsWildCard();
            var is_op_call = IsCall(IsWildCard(), lhs, rhs);
            Const x = (Const)1;
            Const y = (Const)2;
            var z1 = x + y;
            var z2 = x * y;
            Assert.True(is_op_call.MatchLeaf(z1));
            Assert.True(is_op_call.Target.MatchLeaf(z2.Target));

            var is_op_call2 = IsCall(IsWildCard(), IsVArgs(lhs, rhs));

            Assert.IsType<WildCardPattern>(is_op_call2.Parameters[0]);
            Assert.IsType<WildCardPattern>(is_op_call2.Parameters[1]);
        }

        [Fact]
        public void TestTypePattern()
        {
            var ttype = new TensorType(DataType.Float32, new[] { 10, 10 });
            var ty_pat = HasType(ttype);
            Assert.IsType<TypePattern>(ty_pat);
            Assert.True(ty_pat.MatchLeaf(ttype));
        }

        [Fact]
        public void TestDataTypePattern()
        {
            var ttype1 = new TensorType(DataType.Float32, new[] { 10, 10 });
            var ttype2 = new TensorType(DataType.Int16, new[] { 10 });
            var ty_pat = HasDType(DataType.Float32);
            Assert.IsType<TypePattern>(ty_pat);
            Assert.True(ty_pat.MatchLeaf(ttype1));
            Assert.False(ty_pat.MatchLeaf(ttype2));
        }

        [Fact]
        public void TestShapePattern()
        {
            var shape = new int[] { 10, 10 };
            var sp = HasShape(shape);
            var ttype1 = new TensorType(DataType.Float32, new[] { 10, 10 });
            var ttype2 = new TensorType(DataType.Int16, new[] { 10 });
            Assert.True(sp.MatchLeaf(ttype1));
            Assert.False(sp.MatchLeaf(ttype2));
        }

    }

    public class UnitTestGraphMatch : IDisposable
    {

        private readonly EGraph eGraph;
        public UnitTestGraphMatch()
        {
            eGraph = new EGraph();
        }

        public void Dispose()
        {
        }

        [Fact]
        public void TestWildCardRecursion()
        {
            eGraph.Clear();
            WildCardPattern wcx = "x", wcy = "y";

            var pat = wcx + (wcy + IsConst(1));

            Var x = "x", y = "y";
            Expr expr = x + (y + 1);

            eGraph.Add(expr);

            var res = EGraphMatcher.EMatch(eGraph, pat);

            Assert.Single(res);
            Assert.Contains(wcx, res[0].Context.Keys);
            Assert.Contains(wcy, res[0].Context.Keys);
        }

        [Fact]
        public void TestWildCardRecursion2()
        {
            eGraph.Clear();
            WildCardPattern wcx = "x", wcy = "y";
            var pat = wcx + (wcy + IsConst(1));

            Var x = "x", y = "y";
            Expr expr = x + (y + 1);

            eGraph.Add(expr);

            var res = EGraphMatcher.EMatch(eGraph, pat);

            Assert.Single(res);
            Assert.Contains(wcx, res[0].Context.Keys);
            Assert.Contains(wcy, res[0].Context.Keys);
        }

        [Fact]
        public void TestWildCardRecursionWithVArgs()
        {

        }

        [Fact]
        public void TestMatchOpAdd()
        {
            var wc1 = IsWildCard();
            var pat = wc1 + 1;

            var a = new Var("a");
            var wce1 = a * 100 - 32 / 320;
            var e = wce1 + 1;
            var g = new EGraph();
            g.Add(e);

            var matchs = EMatch(g, pat);
            Assert.Single(matchs);
            var result = matchs[0];
            Assert.Contains(wc1, result.Context.Keys);
            Assert.Equal(result[wc1], wce1);
        }

        [Fact]
        public void TestMatchOpOR()
        {
            eGraph.Clear();
            var x = new Var("a");
            var y = x + 10;
            var y1 = y - 10;

            var px = IsWildCard();
            var py = IsBinary(op => op is (BinaryOp.Add or BinaryOp.Sub), px, 10);


            eGraph.Add(y);
            var matchs = EMatch(eGraph, py);
            Assert.Single(matchs);
            eGraph.Add(y1);

            var matchs2 = EMatch(eGraph, py);
            Assert.Equal(2, matchs2.Count);

            var py1 = IsUnary(UnaryOp.Abs, px);
            Assert.Empty(EMatch(eGraph, py1));
        }

        [Fact]
        public void MatchFunction()
        {
            eGraph.Clear();
            Var x = "x";
            Var y = "y";

            WildCardPattern wc1 = "x";
            WildCardPattern wc2 = "y";

            Expr func = new Function(x + y - 1200, x, y);
            ExprPattern pat_1 = new FunctionPattern(x + y - 1200, wc1, wc2);

            ExprPattern pat_2 = new FunctionPattern(x - y, wc1, wc2);

            eGraph.Add(func);
            var res_1 = EMatch(eGraph, pat_1);
            Assert.Single(res_1);

            Assert.Contains(wc1, res_1[0].Context.Keys);
            Assert.Contains(wc2, res_1[0].Context.Keys);

            var res_2 = EMatch(eGraph, pat_2);
            Assert.Empty(res_2);
        }

        [Fact]
        public void TestMatchVArgs()
        {
            eGraph.Clear();

            WildCardPattern wc = "x";
            List<ExprPattern> wcs = new();

            var nest_tuple = new IR.Tuple(4, 5, 6);
            var tuple = new IR.Tuple(1, nest_tuple, 3);
            Expr expr = Concat(tuple, 0);


            ExprPattern vpat = Concat(IsTuple(IsVArgsRepeat((n, param) =>
            {
                for (int i = 0; i < n; i++)
                {
                    var wc = IsWildCard();
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

            eGraph.Add(expr);

            var eMatches = EMatch(eGraph, vpat);
            Assert.Single(eMatches);
            var eMatch = eMatches[0];
            Assert.Equal(eMatch[wcs[0]], tuple[0]);
            Assert.Equal(eMatch[wcs[1]], tuple[1]);
            Assert.Equal(eMatch[wcs[2]], tuple[2]);
        }

        [Fact]
        public void TestMatchVArgsTwice()
        {
            eGraph.Clear();

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

            eGraph.Add(expr);

            var eMatches = EMatch(eGraph, vpat);
            Assert.Single(eMatches);
            var eMatch = eMatches[0];
            Assert.Equal(eMatch[wccons[0]], tuple_rhs[0]);
            Assert.Equal(eMatch[wccons[1]], tuple_rhs[1]);
            Assert.Equal(eMatch[wccons[2]], tuple_rhs[2]);
        }

        [Fact]
        public void TestMatchVArgsRecursion()
        {
            eGraph.Clear();

            Var x = "x";
            Const y = 4;
            Expr z = (Const)1 + 2;

            Const perm = 123;
            Expr expr = Concat(new IR.Tuple(
              Transpose(x, perm),
              Transpose(y, perm),
              Transpose(z, perm)), 0);

            WildCardPattern wc = "wc", wcperm = "perm", wcaxis = "axis";
            List<WildCardPattern> wcin = new();
            var wcvargs = IsVArgsRepeat((n, param) =>
            {
                for (int i = 0; i < n; i++)
                {
                    var input = IsWildCard();
                    param.Add(Transpose(input, wcperm));
                    wcin.Add(input);
                }
            });
            var pattern = Concat(IsTuple(wcvargs), wcaxis);

            eGraph.Add(expr);

            var results = EMatch(eGraph, pattern);
            Assert.Single(results);
            var result = results[0];
            Assert.Equal(result[wcin[0]], x);
            Assert.Equal(result[wcin[1]], y);
            Assert.Equal(result[wcin[2]], z);
            Assert.Equal(result[wcperm], perm);
            Assert.Equal(result[wcaxis], (Const)0);
        }

    }
}
