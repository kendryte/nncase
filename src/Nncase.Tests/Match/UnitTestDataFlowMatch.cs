// using System;
// using Xunit;
// using Nncase.PatternMatch;
// using Nncase.Transform;
// using Nncase.IR;
// using System.Collections.Generic;
// using Nncase.PatternMatch.Math;
// using static Nncase.IR.F.Math;
// using static Nncase.IR.F.Tensors;
// using static Nncase.PatternMatch.Utility;
// using static Nncase.PatternMatch.F.Math;
// using static Nncase.PatternMatch.F.Tensors;
// using static Nncase.IR.TypePatternUtility;

// namespace Nncase.Tests.MatchTest;

// using static Nncase.Transform.DataFlowMatcher;

// public class UnitTestDataFlowMatch
// {
//     [Fact]
//     public void TestMatchDataFlowCallCommutive()
//     {
//         Var x = "x", y = "y";
//         var addpat = IsBinary(BinaryOp.Add, IsVar(), IsVar());
//         Assert.Single(Match(x + y, addpat));
//         Assert.Single(Match(y + x, addpat));
//         var mulpat = IsBinary(BinaryOp.Mul, IsVar(), IsVar());
//         Assert.Single(Match(y * x, mulpat));
//         Assert.Single(Match(x * y, mulpat));
//     }

//     [Fact]
//     public void TestMatchDataFlowNoCallCommutive()
//     {
//         Var x = "x", y = "y";
//         var addpat = IsBinary(BinaryOp.Sub, x, y);
//         Assert.Single(Match(x - y, addpat));
//         Assert.Empty(Match(y - x, addpat));
//         var mulpat = IsBinary(BinaryOp.Div, x, y);
//         Assert.Single(Match(x / y, mulpat));
//         Assert.Empty(Match(y / x, mulpat));
//     }

//     [Fact]
//     public void TestMatchDataFlowCall()
//     {
//         Var x = "x", y = "y";
//         var addpat = IsBinary(BinaryOp.Add, IsWildcard(), IsWildcard());
//         Assert.Single(Match(x + y, addpat));

//         var callpat = IsWildcard();
//         Assert.Single(Match(Square(x), callpat));
//         Assert.Single(Match(x + y, callpat));
//     }

//     [Fact]
//     public void TestNoMatchDataFlowFunc()
//     {
//         Var x = "x", y = "y";
//         var pat = IsBinary(BinaryOp.Add, IsWildcard(), IsWildcard());
//         Assert.Empty(Match(x - y, pat));
//     }

//     [Fact]
//     public void TestMatchDataFlowConst()
//     {
//         Var x = "x", y = "y";
//         var pat = IsBinary(BinaryOp.Sub, IsWildcard(), IsConst());
//         Assert.Single(Match((x + y) - 100, pat));
//     }

//     [Fact]
//     public void TestMatchDataFlowTuple()
//     {
//         Var x = "x", y = "y";
//         var z = x + y;
//         var tuple = new IR.Tuple(x, y, z);
//         var tuplepat = IsTuple(IsVar(), IsWildcard(), IsBinary(BinaryOp.Add, IsWildcard(), IsWildcard()));

//         Assert.Single(Match(tuple, tuplepat));

//         var tuplepat2 = IsTuple();
//         Assert.Single(Match(tuple, tuplepat2));
//     }

//     [Fact]
//     public void TestNotMatchFoldConstCall()
//     {
//         var rule = new Transform.Rule.FoldConstCall();
//         Var x = "x";
//         var z = x + 1;
//         Assert.Empty(Match(z, rule.Pattern));
//     }

//     [Fact]
//     public void TestMatchFoldConstCallTwice()
//     {
//         var rule = new Transform.Rule.FoldConstCall();

//         var z = Concat(new IR.Tuple((Const)2, (Const)1, (Const)2), 0);
//         Assert.Single(Match(z, rule.Pattern));
//         rule.Pattern.Clear();
//         Assert.Single(Match(z, rule.Pattern));
//     }

//     [Fact]
//     public void TestMatchFoldConstCallTwiceFalse()
//     {
//         var rule = new Transform.Rule.FoldConstCall();

//         var z = Concat(new IR.Tuple((Var)"x", (Const)1, (Const)2), 0);
//         Assert.Empty(Match(z, rule.Pattern));

//         rule.Pattern.Clear();
//         var z1 = Concat(new IR.Tuple((Const)4.0f, (Const)1.0f, (Const)1, (Const)2), 0);
//         Assert.Single(Match(z1, rule.Pattern));
//     }
// }
