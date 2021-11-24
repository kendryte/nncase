using System;
using Xunit;
using Nncase.Pattern;
using Nncase.Transform;
using Nncase.IR;
using System.Collections.Generic;
using System.Linq;
using System.Linq.Expressions;
using NetFabric.Hyperlinq;
using Nncase.Pattern.Math;
using static Nncase.IR.F.Math;
using static Nncase.IR.F.Tensors;
using static Nncase.Pattern.Utility;
using static Nncase.Pattern.F.Math;
using static Nncase.Pattern.F.Tensors;
using static Nncase.IR.Utility;
using TorchSharp;
using Nncase.Evaluator;
using Nncase.Importer.TFLite;
using Nncase.IR.F;
using Nncase.IR.NN;
using Binary = Nncase.IR.Math.Binary;

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

        [Fact]
        public void TestFoldConstCallNotMatch()
        {
            var input = new Var("input", new TensorType(DataType.Int32, new Shape(new[] {1, 240, 320, 3})));
            var weights = Const.FromSpan<int>(Enumerable.Range(0, 3 * 3 * 3 * 16).ToArray(), new Shape(new[] {16, 3, 3, 3}));
            var bias = Const.FromSpan<int>(Enumerable.Range(0, 16).ToArray());
            var (inH, inW) = Util.GetHW(input);
            var (fH, fW) = Util.GetHW(weights);
            var strideH = 1;
            var strideW = 1;
            var dilationH = 1;
            var dilationW = 1;
            var padH = TFLiteImporter.GetWindowedPadding(inH, fH, strideH, dilationH, true);
            var padW = TFLiteImporter.GetWindowedPadding(inW, fW, strideW, dilationW, true);
            var stride = Const.FromSpan<int>(new[] { strideH, strideW }, new[] { 2 });
            var dilation = Const.FromSpan<int>(new[] { dilationH, dilationW }, new[] { 2 });
            var padding = Util.ConcatPadding(padH, padW);
            Assert.True(TypeInference.InferenceType(stride));

            var conv = NN.Conv2D(NHWCToNCHW(input), NHWCToNCHW(weights), bias, stride, padding,
                dilation,
                PadMode.Constant, 1);
            var tr = NCHWToNHWC(Clamp(conv, 0, 1));
            var bn = Binary(BinaryOp.Mul, 1, conv);
            Assert.True(TypeInference.InferenceType(conv));
            Assert.True(TypeInference.InferenceType(tr));
            Assert.True(TypeInference.InferenceType(bn));
            var post = DataFlowRewrite.Rewrite(bn,
                new Transform.DataFlow.Rules.FoldConstCall()
            );
            Assert.True(bn.Equals(post));
            Assert.Equal(bn, post);
        }
    }
}