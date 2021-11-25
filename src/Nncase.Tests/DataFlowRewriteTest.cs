using System;
using Xunit;
using Nncase.Pattern;
using Nncase.Transform;
using Nncase.IR;
using System.Collections.Generic;
using System.IO;
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

        public Expr ApplyFoldConstCallRewrite(Expr expr) =>
            DataFlowRewrite.Rewrite(expr, new Transform.DataFlow.Rules.FoldConstCall());
        [Fact]
        public void TestFoldConstCall()
        {
            var lhs = torch.rand(2, 1, 3, torch.ScalarType.Float32);
            var rhs = torch.rand(2, 6, 3, torch.ScalarType.Float32);
            var pre = lhs.ToConst() + rhs.ToConst();
            Assert.True(TypeInference.InferenceType(pre));
            var post = ApplyFoldConstCallRewrite(pre);
            Assert.Equal(lhs + rhs, Evaluator.Evaluator.Eval(post));
        }

        [Fact]
        public void TestFoldConstCallTuple()
        {
            var lhs = torch.rand(2, 1, 3, torch.ScalarType.Float32);
            var rhs = torch.rand(2, 6, 3, torch.ScalarType.Float32);
            var pre = Concat(new IR.Tuple(lhs.ToConst(), rhs.ToConst()), 1);
            Assert.True(TypeInference.InferenceType(pre));
            var post = ApplyFoldConstCallRewrite(pre);
            Assert.IsType<Const>(post);
            Assert.Equal(torch.cat(new[] { lhs, rhs }, 1), Evaluator.Evaluator.Eval(post));
        }

        [Fact]
        public void TestFoldConstCallType()
        {
            var a = (Const) 1;
            var b = (Const) 2;
            var expr = a * b + 3;
            Assert.True(TypeInference.InferenceType(expr));
            var post = ApplyFoldConstCallRewrite(expr);
            Assert.True(TypeInference.InferenceType(post));
            Assert.Equal(expr.CheckedType, post.CheckedType);
            Assert.Equal((post as Const).ToScalar<int>(), 1 * 2 + 3);
        }

        [Fact]
        public void TestComplexFoldConstCall()
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

            
            TypeInference.InferenceType(padding);
            var fold_padding = DataFlowRewrite.Rewrite(padding,
                new Transform.DataFlow.Rules.FoldConstCall()
            );
            TypeInference.InferenceType(fold_padding);
            var old_conv = NN.Conv2D(NHWCToNCHW(input), NHWCToNCHW(weights), bias, stride, padding,
                dilation,
                PadMode.Constant, 1);

            TypeInference.InferenceType(old_conv);
            
            var conv = DataFlowRewrite.Rewrite(old_conv,
                new Transform.DataFlow.Rules.FoldConstCall()
            );
            var tr = NCHWToNHWC(Clamp(conv, 0, 1));
            var bn = Binary(BinaryOp.Mul, 1, tr);
            var max = Binary(BinaryOp.Max, conv, bn);
            Assert.True(TypeInference.InferenceType(bn));
            var doubleV = Const.FromSpan<int>(new[] { 2, 2 }, new[] { 2 });
            
            var initValue = (Const) 0;
            var (rInH, rInW) = Util.GetHW(max);
            var rPadH = TFLiteImporter.GetWindowedPadding(rInH, 2, 2, dilationH, true);
            var rPadW = TFLiteImporter.GetWindowedPadding(rInW, 2, 2, dilationW, true);
            var rPadding = Util.ConcatPadding(rPadH, rPadW);

            var reduce = ReduceWindow2D(ReduceOp.Max, max, initValue, doubleV, doubleV, rPadding, dilation);
            Assert.True(TypeInference.InferenceType(max));
            Assert.True(TypeInference.InferenceType(initValue));
            var post = DataFlowRewrite.Rewrite(bn,
                new Transform.DataFlow.Rules.FoldConstCall()
            );
            Assert.True(TypeInference.InferenceType(post));
        }
    }
}