using System;
using Xunit;
using Nncase.Pattern;
using Nncase.Transform;
using Nncase.IR;
using System.Collections.Generic;
using System.Linq.Expressions;
using Nncase.Pattern.Math;
using static Nncase.IR.F.Math;
using static Nncase.IR.F.Tensors;
using static Nncase.Pattern.Utility;
using static Nncase.Pattern.F.Math;
using static Nncase.Pattern.F.Tensors;
using static Nncase.IR.Utility;
using TorchSharp;
using Nncase.Evaluator;

namespace Nncase.Tests
{

    internal class SwapXY : PatternRule
    {
        BinaryWrapper binary;
        public SwapXY()
        {
            Pattern = binary = Add(IsWildCard(), IsConst());
        }
        public override Expr GetRePlace(IMatchResult result)
        {
            binary.Bind(result);
            return binary.Rhs() + binary.Lhs();
        }
    }
    
    public class UnitTestExpressionsRewrite
    {

        [Fact]
        public void TestGetExpressions()
        {
            // Expression dd = (ConstPattern x, ExprPattern y) => x + y;
        }

    }

    public class UnitTestDataFlowRewrite
    {
        [Fact]
        public void TestSwapXY()
        {
            var lhs = (Var)"x";
            var rhs = (Const)1;
            var pre = (lhs + rhs) * 10;
            var post = DataFlowRewrite.Rewrite(pre, new SwapXY());
            Assert.Equal(post, (rhs + lhs) * 10);
        }

        [Fact]
        public void TestRemoveShapeOp()
        {
            var lhs = new Var("x", new TensorType(DataType.Float32, new[] { 1, 1, 3 }));
            var rhs = torch.rand(1, 6, 3, torch.ScalarType.Float32).ToConst();
            var pre = ShapeOp(lhs + rhs) + 1;
            TypeInference.InferenceType(pre);
            var post = DataFlowRewrite.Rewrite(pre, new Transform.DataFlow.Rules.FoldShapeOp());
            Assert.Equal(torch.tensor(new[] { 2, 7, 4 }, torch.ScalarType.Int32), Evaluator.Evaluator.Eval(post));
        }

        [Fact]
        public void TestFoldConstCall()
        {
            var lhs = torch.rand(2, 1, 3, torch.ScalarType.Float32);
            var rhs = torch.rand(2, 6, 3, torch.ScalarType.Float32);
            var pre = lhs.ToConst() + rhs.ToConst();
            Assert.True(TypeInference.InferenceType(pre));
            var post = DataFlowRewrite.Rewrite(pre, new Transform.DataFlow.Rules.FoldConstCall());
            Assert.Equal(lhs + rhs, Evaluator.Evaluator.Eval(post));
        }

        [Fact]
        public void TestFoldConstCallTuple()
        {
            var lhs = torch.rand(2, 1, 3, torch.ScalarType.Float32);
            var rhs = torch.rand(2, 6, 3, torch.ScalarType.Float32);
            var pre = Concat(new IR.Tuple(lhs.ToConst(), rhs.ToConst()), 1);
            Assert.True(TypeInference.InferenceType(pre));
            var post = DataFlowRewrite.Rewrite(pre,
             new Transform.DataFlow.Rules.FoldConstCall()
            );
            Assert.IsType<Const>(post);
            Assert.Equal(torch.cat(new[] { lhs, rhs }, 1), Evaluator.Evaluator.Eval(post));
        }
    }
}