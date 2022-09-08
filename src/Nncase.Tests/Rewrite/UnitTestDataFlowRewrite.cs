using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Numerics.Tensors;
using System.Threading.Tasks;
using Microsoft.Extensions.Hosting;
using NetFabric.Hyperlinq;
using Nncase.Evaluator;
using Nncase.Importer;
using Nncase.Importer.TFLite;
using Nncase.IR;
using Nncase.IR.F;
using Nncase.IR.NN;
using Nncase.PatternMatch;
using Nncase.Tests.RewriteTest;
using Nncase.Transform;
using Nncase.Transform.Passes;
using OrtKISharp;
using Xunit;
using static Nncase.IR.F.Math;
using static Nncase.IR.F.NN;
using static Nncase.IR.F.Tensors;
using static Nncase.IR.TypePatternUtility;
using static Nncase.PatternMatch.F.Math;
using static Nncase.PatternMatch.F.Tensors;
using static Nncase.PatternMatch.Utility;
using Binary = Nncase.IR.Math.Binary;
using Broadcast = Nncase.IR.Tensors.Broadcast;
using Tuple = Nncase.IR.Tuple;

namespace Nncase.Tests.RewriteTest
{

    public class DataFlowRewriteTestFactory : RewriteFixtrue
    {
        public DataFlowRewriteTestFactory() : base()
        {
        }

        private static IEnumerable<object[]> Data =>
          new List<object[]>
          {
             new object[] {},
          };

        // [Theory]
        // [MemberData(nameof(DataOne))]
        // public void RunOne(IRewriteCase Case) => RunCore(Case);

        protected void RunCore(IRewriteCase Case)
        {
            var caseOptions = GetPassOptions();
            Expr pre = Case.PreExpr;
            var infered = pre.InferenceType();
            CompilerServices.DumpIR(pre, "pre", caseOptions.PassDumpDir);
            Assert.True(infered);
            var post = CompilerServices.Rewrite(pre, Case.Rules, caseOptions);
            Assert.True(post.InferenceType());
            CompilerServices.DumpIR(post, "post", caseOptions.PassDumpDir);
            Assert.Equal(Case.PostExpr, post);
        }

        // [Theory]
        // [MemberData(nameof(DataAll))]
        // public void RunAll(IRewriteCase Case) => RunCore(Case);


        public static IEnumerable<object[]> DataOne => Data.Take(1);
        public static IEnumerable<object[]> DataAll => Data.Skip(1);
    }



    // internal class SwapXY : IRewriteRule
    // {
    //     BinaryWrapper binary;
    //     public SwapXY()
    //     {
    //         Pattern = binary = Add(IsWildcard(), IsConst());
    //     }
    //     public override Expr GetReplace(IMatchResult result)
    //     {
    //         binary.Bind(result);
    //         return binary.Rhs() + binary.Lhs();
    //     }
    // }

    public class UnitTestDataFlowRewrite : RewriteFixtrue
    {
        public UnitTestDataFlowRewrite() : base()
        {
        }

        // [Fact]
        // public void TestSwapXY()
        // {
        //     var lhs = (Var)"x";
        //     var rhs = (Const)1;
        //     var pre = (lhs + rhs) * 10;

        //     var post = CompilerServices.Rewrite(pre, new[] { new SwapXY() }, RunPassOptions.Invalid);
        //     Assert.Equal(post, (rhs + lhs) * 10);
        // }

        // [Fact]
        // public void TestRemoveShapeOp()
        // {
        //     var lhs = new Var("x", new TensorType(DataTypes.Float32, new[] { 1, 1, 3 }));
        //     var rhs = torch.rand(1, 6, 3, torch.ScalarType.Float32).ToTensor();
        //     var pre = ShapeOf(lhs + rhs);
        //     Assert.True(CompilerServices.InferenceType(pre));
        //     var post = CompilerServices.Rewrite(pre, new[] { new Transform.Rule.FoldShapeOp() }, RunPassOptions.Invalid);
        //     Assert.Equal(new[] { 1, 6, 3 }, ((TensorConst)post).Value.Cast<int>().ToArray());
        // }

        [Fact]
        public void TestFoldConstCall()
        {
            var caseOptions = GetPassOptions();
            var lhs = OrtKI.Random(2, 1, 3);
            var rhs = OrtKI.Random(2, 6, 3);
            var pre = (Const)lhs.ToTensor() + rhs.ToTensor();
            Assert.True(CompilerServices.InferenceType(pre));
            var post = ApplyFoldConstCallRewrite(pre, caseOptions);
            Assert.Equal(lhs + rhs, post.Evaluate().AsTensor().ToOrtTensor());
        }

        [Fact]
        public void TestFoldConstCallTuple()
        {
            var caseOptions = GetPassOptions();
            var lhs = OrtKI.Random(2, 1, 3);
            var rhs = OrtKI.Random(2, 6, 3);
            var pre = Concat(new IR.Tuple(lhs.ToTensor(), rhs.ToTensor()), 1);
            Assert.True(CompilerServices.InferenceType(pre));
            var post = ApplyFoldConstCallRewrite(pre, caseOptions);
            Assert.IsType<TensorConst>(post);
            Assert.Equal(OrtKI.Concat(new[] { lhs, rhs }, 1), post.Evaluate().AsTensor().ToOrtTensor());
        }

        [Fact]
        public void TestFoldConstCallType()
        {
            var caseOptions = GetPassOptions();
            var a = (Const)1;
            var b = (Const)2;
            var expr = a * b + 3;
            Assert.True(CompilerServices.InferenceType(expr));
            var post = ApplyFoldConstCallRewrite(expr, caseOptions);
            Assert.True(CompilerServices.InferenceType(post));
            Assert.Equal(expr.CheckedType, post.CheckedType);
            var res = 1 * 2 + 3;
            Assert.Equal(((TensorConst)post).Value.ToScalar<int>(), res);

            var cast_to_i64 = Cast(expr, DataTypes.Int64);
            Assert.True(CompilerServices.InferenceType(cast_to_i64));

            var cast_to_i32 = Cast(cast_to_i64, DataTypes.Int32);
            Assert.True(CompilerServices.InferenceType(cast_to_i32));

            var cat = Stack(new Tuple(cast_to_i32, cast_to_i32), 0);
            Assert.True(CompilerServices.InferenceType(cat));
            var old_dtype = cat.CheckedDataType;
            var after_cat = ApplyFoldConstCallRewrite(cat, caseOptions);

            Assert.Equal(
                (after_cat as TensorConst).Value.Cast<int>().ToArray(),
                new[] { res, res });
            Assert.Equal(old_dtype, after_cat.CheckedDataType);
        }

        // [Fact]
        // public void TestRewriteSameAsShapeInferPass()
        // {
        //     GetPassOptions();
        //     var input = new Var("input", new TensorType(DataTypes.Int32, new Shape(new[] { 1, 3, 240, 320 })));
        //     Assert.True(CompilerServices.InferenceType(input));
        //     var computeShape = ShapeOf(input);
        //     var shapeRewrite = CompilerServices.Rewrite(computeShape,
        //         new IRewriteRule[] { new Transform.Rule.FoldShapeOp() }, RunPassOptions.Invalid);
        //     var shapePass = RunShapeInferPass("", computeShape, input);
        //     Assert.Equal(shapeRewrite, shapePass);
        // }

        [Fact]
        public async Task TestFoldExpand()
        {
            var caseOptions = GetPassOptions();
            var weights = new Var("weights", new TensorType(DataTypes.Float32, new Shape(1, 3, 224, 224)));
            var t = Util.ShapeIndex(weights, 0);
            var expand = Expand(0f, Cast(Util.ShapeIndex(weights, 0), DataTypes.Int64));
            var s = await RunShapeInferPass("", caseOptions, expand, weights);
            Assert.True(s is Const);
        }

        [Fact]
        public async Task TestFoldShapeOf()
        {
            var caseOptions = GetPassOptions();
            var input = new Var("input", new TensorType(DataTypes.Int32, new Shape(1, 3, 240, 320)));
            var shape = ShapeOf(input);
            var shapePost = await RunShapeInferPass("FoldShapeOf", caseOptions, shape);
            Assert.Equal(new long[] { 1, 3, 240, 320 }, ((TensorConst)shapePost).Value.ToArray<long>());
        }

        [Fact]
        public async Task TestExpandToRank()
        {
            var caseOptions = GetPassOptions();
            var input = new Var("input", new TensorType(DataTypes.Int32, new Shape(1, 3, 240, 320)));
            var exp = Expand(1, Cast(Rank(input) - 0, new Int64Type()));
            var result = await RunShapeInferPass("ExpandToRank", caseOptions, exp);
            Assert.Equal(new[] { 1, 1, 1, 1 }, result.Evaluate().AsTensor().ToArray<int>());
        }
    }

    public class UnitTestDataFlowRewriteAndInferIntegrate : RewriteFixtrue
    {
        public UnitTestDataFlowRewriteAndInferIntegrate() : base()
        {
        }

        public T Dim1ExprToScalar<T>(Expr expr) where T : unmanaged, System.IEquatable<T> => (expr as TensorConst).Value.Cast<T>()[0];

        [Fact]
        public async Task TestPaddingCompute()
        {
            var caseOptions = GetPassOptions();
            var input = new Var("input", new TensorType(DataTypes.Int32, new Shape(1, 3, 33, 65)));
            var weights = Tensor.FromSpan<int>(Enumerable.Range(0, 3 * 3 * 3 * 16).ToArray(), new Shape(new[] { 16, 3, 3, 3 }));
            var (inH, inW) = Util.GetHW(input);
            var (fH, fW) = Util.GetHW(weights);
            var inHPost = await RunShapeInferPass("inH", caseOptions, inH);
            var inWPost = await RunShapeInferPass("inW", caseOptions, inW);
            Assert.Equal(33, ((TensorConst)inHPost).Value.ToScalar<int>());
            Assert.Equal(65, ((TensorConst)inWPost).Value.ToScalar<int>());
            var strideH = 1;
            var strideW = 1;
            var dilationH = 1;
            var dilationW = 1;
            var padH = Util.GetWindowedPadding(inH, fH, strideH, dilationH, true);
            var padW = Util.GetWindowedPadding(inW, fW, strideW, dilationW, true);
            var padding = Util.ConcatPadding(padH, padW);
            // Assert.True(CompilerServices.InferenceType(padding));
            var paddingPost = await RunShapeInferPass("padding", caseOptions, padding, input);
            Assert.Equal(Tensor.FromSpan(new[] { 1, 1, 1, 1 }, new Shape(2, 2)), paddingPost);
        }

        [Fact]
        public async Task TestYolo20MinStructure()
        {
            var caseOptions = GetPassOptions();
            var input = new Var("input", new TensorType(DataTypes.Int32, new Shape(new[] { 1, 240, 320, 3 })));
            var weights = Tensor.FromSpan<int>(Enumerable.Range(0, 3 * 3 * 3 * 16).ToArray(), new Shape(new[] { 16, 3, 3, 3 }));
            var bias = Tensor.FromSpan<int>(Enumerable.Range(0, 16).ToArray());
            var (inH, inW) = Util.GetHW(input);
            var (fH, fW) = Util.GetHW(weights);
            var strideH = 1;
            var strideW = 1;
            var dilationH = 1;
            var dilationW = 1;
            var padH = Util.GetWindowedPadding(inH, fH, strideH, dilationH, true);
            var padW = Util.GetWindowedPadding(inW, fW, strideW, dilationW, true);
            var stride = Tensor.FromSpan<int>(new[] { strideH, strideW }, new[] { 2 });
            var dilation = Tensor.FromSpan<int>(new[] { dilationH, dilationW }, new[] { 2 });
            var padding = Util.ConcatPadding(padH, padW);

            var conv = NN.Conv2D(NHWCToNCHW(input), NHWCToNCHW(weights), bias, stride, padding,
                dilation,
                PadMode.Constant, 1);
            var convAfterTranspose = NCHWToNHWC(Clamp(conv, 0, 1));

            var postConvAfterTranspose = await RunShapeInferPass("convAfterTranspose", caseOptions, convAfterTranspose);
            Assert.True(CompilerServices.InferenceType(postConvAfterTranspose));
            Assert.Equal(new Shape(1, 240, 320, 16), postConvAfterTranspose.CheckedShape);

            var mul = Binary(BinaryOp.Mul, 1, convAfterTranspose);
            var max = Binary(BinaryOp.Max, convAfterTranspose, mul);

            // ReduceWindow2D
            var doubleV = Tensor.FromSpan<int>(new[] { 2, 2 }, new[] { 2 });
            var initValue = (Const)0;
            var (rInH, rInW) = Util.GetHW(max);
            var rPadH = Util.GetWindowedPadding(rInH, 2, 2, dilationH, true);
            var rPadW = Util.GetWindowedPadding(rInW, 2, 2, dilationW, true);
            var rPadding = Util.ConcatPadding(rPadH, rPadW);
            var reduce = NCHWToNHWC(ReduceWindow2D(ReduceOp.Max, NHWCToNCHW(max), initValue, doubleV, doubleV, rPadding, dilation, false, false));
            var post = await RunShapeInferPass("reduce", caseOptions, reduce);
            Assert.True(CompilerServices.InferenceType(post));
            Assert.Equal(new Shape(1, 120, 160, 16), post.CheckedShape);
        }

        [Fact]
        public async Task SliceForShapeIndex()
        {
            var caseOptions = GetPassOptions();
            var input = new Var(new TensorType(DataTypes.Float32, new Shape(1, 7, 7, 75)));
            var slice = Util.ShapeIndex(input, 1);
            CompilerServices.InferenceType(slice);
            var post = await RunShapeInferPass("slice", caseOptions, slice);
            Assert.True(CompilerServices.InferenceType(post));
            Assert.True(post is Const);
            Assert.Equal(Shape.Scalar, post.CheckedShape);
        }

        [Fact]
        public async Task SoftMaxImporterProcess()
        {
            var caseOptions = GetPassOptions();
            var input = new Var(new TensorType(DataTypes.Float32, new Shape(1, 3, 224, 224)));
            var axis = -1;
            var inShape = ShapeOf(input);
            Expr axisExprBefore = axis < 0
                ? axis + Rank(input)
                : Tensor.FromSpan<int>(new[] { axis });
            axisExprBefore.InferenceType();
            var axisExpr = await RunShapeInferPass("Axis", caseOptions, axisExprBefore, input);
            Assert.Equal(3, ((TensorConst)axisExpr).Value.Cast<int>()[0]);
            var firstSliceBefore = Slice(inShape, new[] { 0 }, axisExpr, 1);
            firstSliceBefore.InferenceType();
            var firstSlice = await RunShapeInferPass("firstSlice", caseOptions, firstSliceBefore, input);
            Assert.Equal(new[] { 1, 3, 224 }, ((TensorConst)firstSlice).Value.ToArray<int>());
            var firstSizeBefore = Prod(firstSlice);
            firstSizeBefore.InferenceType();
            var firstSize = await RunShapeInferPass("firstSize", caseOptions, firstSizeBefore, input);
            Assert.Equal(1 * 3 * 224, ((TensorConst)firstSize).Value.ToScalar<int>());
            var secondBefore = Prod(Slice(inShape, axisExpr, Rank(input), 1));
            var secondSize = await RunShapeInferPass("secondSize", caseOptions, secondBefore, input);
            Assert.Equal(224, ((TensorConst)secondSize).Value.ToScalar<int>());
            var beforeShape = Concat(new Tuple(firstSize, secondSize), 0);
            var afterShape = ShapeOf(input);
            var softMax = Reshape(
                NN.Softmax(
                    Reshape(input, beforeShape),
                    axis),
                afterShape);
            Assert.True(softMax.InferenceType());
        }

        [Fact]
        public async Task TestReshapeToByChannel()
        {
            var caseOptions = GetPassOptions();
            var v = Tensor.FromSpan<int>(new[] { 1, 2, 3 });
            var shape = Concat(
                new IR.Tuple(
                    ShapeOf(v),
                    new long[] { 1 },
                    new long[] { 1 }), 0);
            var afterShape = await RunShapeInferPass("Shape", caseOptions, shape);
            Assert.True(afterShape.InferenceType());
            Assert.Equal(new long[] { 3, 1, 1 }, afterShape);
            var b = Reshape(v, afterShape);
            b.InferenceType();
            Assert.Equal(new[] { 3, 1, 1 }, b.Evaluate().AsTensor().Dimensions.ToArray());

            var a = OnnxImporter.ReshapeToByChannel(v);
            var after = await RunShapeInferPass("ReshapeToByChannel", caseOptions, a);
            Assert.True(after.InferenceType());
            Assert.Equal(new[] { 3, 1, 1 }, after.Evaluate().AsTensor().Dimensions.ToArray());
        }
    }
}