using System;
using Xunit;
using Nncase.Pattern;
using Nncase.Transform;
using Nncase.IR;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Linq.Expressions;
using System.Numerics.Tensors;
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
using Nncase.Tests.ReWrite;
using Nncase.Transform.DataFlow.Rules;
using Nncase.Transform.Rule;
using Binary = Nncase.IR.Math.Binary;
using Tuple = Nncase.IR.Tuple;

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

    public class UnitTestDataFlowRewrite : RewriteTest
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
            var pre = ShapeOp(lhs + rhs);
            Assert.True(TypeInference.InferenceType(pre));
            var post = DataFlowRewrite.Rewrite(pre, new Transform.DataFlow.Rules.FoldShapeOp());
            Assert.Equal(new[] { 1, 6, 3 }, post.ToTensor<int>().ToArray());
        }

        [Fact]
        public void TestFoldConstCall()
        {
            var lhs = torch.rand(2, 1, 3, torch.ScalarType.Float32);
            var rhs = torch.rand(2, 6, 3, torch.ScalarType.Float32);
            var pre = lhs.ToConst() + rhs.ToConst();
            Assert.True(TypeInference.InferenceType(pre));
            var post = ApplyFoldConstCallRewrite(pre);
            Assert.Equal(lhs + rhs, post.Eval());
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
            var res = 1 * 2 + 3;
            Assert.Equal(post.ToScalar<int>(), res);
            
            var cast_to_i64 = Cast(expr, DataType.Int64);
            Assert.True(TypeInference.InferenceType(cast_to_i64));
            
            var cast_to_i32 = Cast(cast_to_i64, DataType.Int32);
            Assert.True(TypeInference.InferenceType(cast_to_i32));
            
            var cat = Stack(new Tuple(cast_to_i32, cast_to_i32), 0);
            Assert.True(TypeInference.InferenceType(cat));
            var old_dtype = cat.CheckedDataType;
            var after_cat = ApplyFoldConstCallRewrite(cat);
            
            Assert.Equal(
                (after_cat as Const).ToTensor<int>().ToArray(),
                new[] {res, res});
            Assert.Equal(old_dtype, after_cat.CheckedDataType);
        }

        [Fact]
        public void TestRewriteSameAsShapeInferPass()
        {
            var input = new Var("input", new TensorType(DataType.Int32, new Shape(new[] { 1, 3, 240, 320 })));
            Assert.True(TypeInference.InferenceType(input));
            var computeShape = ShapeOp(input);
            var shapeRewrite = DataFlowRewrite.Rewrite(computeShape,
                new PatternRule[] { new Transform.DataFlow.Rules.FoldShapeOp() });
            var shapePass = RunShapeInferPass(computeShape, input);
            Assert.Equal(shapeRewrite, shapePass);
        }
    }

    public class DataFlowRewriteAndInferIntegrateTest : RewriteTest
    {
        public T Dim1ExprToScalar<T>(Expr expr) where T : unmanaged => (expr as Const).ToTensor<T>().ToArray()[0];

        [Fact]
        public void TestPaddingCompute()
        {
            var input = new Var("input", new TensorType(DataType.Int32, new Shape(1, 3, 240, 320)));
            var weights = Const.FromSpan<int>(Enumerable.Range(0, 3 * 3 * 3 * 16).ToArray(), new Shape(new[] { 16, 3, 3, 3 }));
            var (inH, inW) = Util.GetHW(input);
            var (fH, fW) = Util.GetHW(weights);
            var inHPost = RunShapeInferPass(inH);
            var inWPost = RunShapeInferPass(inW);
            Assert.Equal(240, Dim1ExprToScalar<int>(inHPost));
            Assert.Equal(320, Dim1ExprToScalar<int>(inWPost));
            var strideH = 1;
            var strideW = 1;
            var dilationH = 1;
            var dilationW = 1;
            var padH = TFLiteImporter.GetWindowedPadding(inH, fH, strideH, dilationH, true);
            var padW = TFLiteImporter.GetWindowedPadding(inW, fW, strideW, dilationW, true);
            var padding = Util.ConcatPadding(padH, padW);
            Assert.True(TypeInference.InferenceType(padding));
            var paddingPost = RunShapeInferPass(padding, input);
            Assert.True(paddingPost is Const);
        }

        [Fact]
        public void TestYolo20MinStructure()
        {
            var input = new Var("input", new TensorType(DataType.Int32, new Shape(new[] { 1, 240, 320, 3 })));
            var weights = Const.FromSpan<int>(Enumerable.Range(0, 3 * 3 * 3 * 16).ToArray(), new Shape(new[] { 16, 3, 3, 3 }));
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

            var conv = NN.Conv2D(NHWCToNCHW(input), NHWCToNCHW(weights), bias, stride, padding,
                dilation,
                PadMode.Constant, 1);
            var convAfterTranspose = NCHWToNHWC(Clamp(conv, 0, 1));

            var postConvAfterTranspose = RunShapeInferPass(convAfterTranspose);
            Assert.True(TypeInference.InferenceType(postConvAfterTranspose));
            Assert.Equal(new Shape(1, 240, 320, 16), postConvAfterTranspose.CheckedShape);

            var mul = Binary(BinaryOp.Mul, 1, convAfterTranspose);
            var max = Binary(BinaryOp.Max, convAfterTranspose, mul);
            Assert.True(TypeInference.InferenceType(mul));

            // ReduceWindow2D
            var doubleV = Const.FromSpan<int>(new[] { 2, 2 }, new[] { 2 });
            var initValue = (Const)0;
            var (rInH, rInW) = Util.GetHW(max);
            var rPadH = TFLiteImporter.GetWindowedPadding(rInH, 2, 2, dilationH, true);
            var rPadW = TFLiteImporter.GetWindowedPadding(rInW, 2, 2, dilationW, true);
            var rPadding = Util.ConcatPadding(rPadH, rPadW);
            var reduce = NCHWToNHWC(ReduceWindow2D(ReduceOp.Max, NHWCToNCHW(max), initValue, doubleV, doubleV, rPadding, dilation));
            var post = RunShapeInferPass(reduce);
            Assert.True(TypeInference.InferenceType(post));
            Assert.Equal(new Shape(1, 120, 160, 16), post.CheckedShape);
        }

        [Fact]
        public void SliceShapeOp()
        {
            var input = new Var(new TensorType(DataType.Float32, new Shape(1, 7, 7, 75)));
            var slice = Util.ShapeIndex(ShapeOp(input), 1);
            TypeInference.InferenceType(slice);
            var post = RunShapeInferPass(slice);
            Assert.True(TypeInference.InferenceType(post));
            Assert.True(post is Const);
            Assert.Equal(Shape.Scalar, post.CheckedShape);
        }
    }
}