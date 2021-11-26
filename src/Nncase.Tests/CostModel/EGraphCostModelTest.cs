using Nncase.CostModel;
using Nncase.IR;
using Nncase.IR.F;
using Xunit;
using Nncase.Transform;
using Nncase.Pattern;
using static Nncase.Pattern.Utility;
using Nncase.Tests.ReWrite;
using System;

namespace Nncase.Tests.CostModel
{

    public class EGraphCostModelTest : RewriteTest
    {
        [Fact]
        public void TestConst()
        {
            var expr = (Const)1 + ((Const)2 * ((Const)3 / (Const)5));
            TypeInference.InferenceType(expr);
            var egraph = new EGraph(expr);
            var graphCosts = egraph.Costs();
            Assert.Equal(new Cost(0, 0), graphCosts[expr.Target]);
            Assert.Equal(new Cost(0, 4), graphCosts[expr.Parameters[0]]);
            Assert.Equal(new Cost(2, 4 * 3), graphCosts[expr.Parameters[1]]);
        }

        [Fact]
        public void TestConstXmul1()
        {
            var lhs = ((Const)2 * ((Const)3 / (Const)5));
            var expr = lhs * (Const)1;
            TypeInference.InferenceType(expr);
            var egraph = new EGraph();
            egraph.Add(expr, out var root);
            EGraphReWriter.ReWrite(egraph, new DemoRuleXmul1(), passOptions.SetName("TestConstXmul1"));
            var new_expr = egraph.Extract(root, passOptions);
            Console.WriteLine(new_expr.DumpExprAsIL());
            Console.WriteLine(lhs.DumpExprAsIL());
            Assert.Equal(lhs, new_expr);
        }
    }
}