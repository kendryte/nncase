// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.
// public class UnitTestEGraphCostModel : RewriteFixtrue
//     {
//         public UnitTestEGraphCostModel(IHost host) : base(host) { }

// [Fact]
//         public void TestConst()
//         {
//             var expr = (Const)1 + ((Const)2 * ((Const)3 / (Const)5));
//             CompilerServices.InferenceType(expr);
//             var egraph = new EGraph(expr);
//             var graphCosts = egraph.Costs();
//             Assert.Equal(new Cost(0, 0), graphCosts[expr.Target]);
//             Assert.Equal(new Cost(0, 4), graphCosts[expr.Parameters[0]]);
//             Assert.Equal(new Cost(2, 4 * 3), graphCosts[expr.Parameters[1]]);
//         }

// [Fact]
//         public void TestConstXmul1()
//         {
//             var lhs = ((Const)2 * ((Const)3 / (Const)5));
//             var expr = lhs * (Const)1;
//             CompilerServices.InferenceType(expr);
//             var egraph = new EGraph();
//             egraph.Add(expr, out var root);
//             EGraphReWriter.ReWrite(egraph, new Transform.Rule.Xmul1(), passOptions.SetName("EGraphCostModelTest/TestConstXmul1"));
//             var new_expr = egraph.Extract(root, passOptions);
//             Console.WriteLine(new_expr.DumpExprAsIL());
//             Console.WriteLine(lhs.DumpExprAsIL());
//             Assert.Equal(lhs, new_expr);
//         }
//     }
// }
