// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.


// public class UnitTestCostModel
// {
//     [Fact]
//     public void TestConst()
//     {
//         var a = (Const)7;
//         var exprVisitor = new ExprCostModelVisitor();
//         Assert.True(a.InferenceType());
//         Assert.Equal(new Cost(0, 4), exprVisitor.Visit(a));
//     }

// [Fact]
//     public void TestBinary()
//     {
//         // todo need process pow lhs is not cost
//         // var a = (Const)1;
//         // var n = (Const)5;
//         // var pow = Math.Pow(a, n);
//         // CompilerServices.InferenceType(pow);
//         // var exprVisitor = new ExprCostModelVisitor();
//         // Assert.Equal(new Cost(5, 0), exprVisitor.Visit(pow));
//     }

// [Fact]
//     public void TestCostInf()
//     {
//         var c = Cost.Inf;
//         Assert.Equal(Cost.Inf, c + new Cost(10, 120));
//     }
// }
