// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using Microsoft.Extensions.DependencyInjection;
using NetFabric.Hyperlinq;
using Nncase.Evaluator;
using Nncase.Importer;
using Nncase.IR;
using Nncase.IR.F;
using Nncase.IR.NN;
using Nncase.IR.Tensors;
using Nncase.Passes;
using Nncase.Passes.Analysis;
using Nncase.PatternMatch;
using Nncase.Tests.TestFixture;
using OrtKISharp;
using Xunit;
using static Nncase.IR.F.Math;
using static Nncase.IR.F.NN;
using static Nncase.IR.F.Tensors;
using static Nncase.PatternMatch.Utility;
using Tuple = Nncase.IR.Tuple;

namespace Nncase.Tests.ReWriteTest;

[AutoSetupTestMethod(InitSession = true)]
public class UnitTestDataFlowRewrite : RewriteFixtrue
{
    [Fact]
    public void TestFoldConstCall()
    {
        var lhs = OrtKI.Random(2, 1, 3);
        var rhs = OrtKI.Random(2, 6, 3);
        var pre = (Const)lhs.ToTensor() + rhs.ToTensor();
        Assert.True(CompilerServices.InferenceType(pre));
        var post = ApplyFoldConstCallRewrite(pre);
        Assert.Equal(lhs + rhs, post.Evaluate().AsTensor().ToOrtTensor());
    }

    [Fact]
    public void TestFoldConstCallTuple()
    {
        var lhs = OrtKI.Random(2, 1, 3);
        var rhs = OrtKI.Random(2, 6, 3);
        var pre = Concat(new IR.Tuple(lhs.ToTensor(), rhs.ToTensor()), 1);
        Assert.True(CompilerServices.InferenceType(pre));
        var post = ApplyFoldConstCallRewrite(pre);
        Assert.IsType<TensorConst>(post);
        Assert.Equal(OrtKI.Concat(new[] { lhs, rhs }, 1), post.Evaluate().AsTensor().ToOrtTensor());
    }

    [Fact]
    public void TestFoldConstCallType()
    {
        var a = (Const)1;
        var b = (Const)2;
        var expr = (a * b) + 3;
        using var exprPin = new ExprPinner(expr);
        var post = ApplyFoldConstCallRewrite(expr);
        Assert.True(CompilerServices.InferenceType(expr));
        Assert.True(CompilerServices.InferenceType(post));
        Assert.Equal(expr.CheckedType, post.CheckedType);
        var res = (1 * 2) + 3;
        Assert.Equal(((TensorConst)post).Value.ToScalar<int>(), res);

        var cast_to_i64 = Cast(expr, DataTypes.Int64);
        Assert.True(CompilerServices.InferenceType(cast_to_i64));

        var cast_to_i32 = Cast(cast_to_i64, DataTypes.Int32);
        Assert.True(CompilerServices.InferenceType(cast_to_i32));

        var cat = Stack(new Tuple(cast_to_i32, cast_to_i32), 0);
        Assert.True(CompilerServices.InferenceType(cat));
        var old_dtype = cat.CheckedDataType;
        var after_cat = ApplyFoldConstCallRewrite(cat);

        Assert.Equal(
            ((TensorConst)after_cat).Value.Cast<int>().ToArray(),
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
    //         new IRewriteRule[] { new Transform.Pass.FoldShapeOp() }, RunPassContext.Invalid);
    //     var shapePass = RunShapeInferPass("", computeShape, input);
    //     Assert.Equal(shapeRewrite, shapePass);
    // }
    [Fact]
    public async Task TestFoldExpand()
    {
        var weights = new Var("weights", new TensorType(DataTypes.Float32, new Shape(1, 3, 224, 224)));
        _ = Util.ShapeIndex(weights, 0);
        var expand = Expand(0f, Cast(Util.ShapeIndex(weights, 0), DataTypes.Int64));
        var s = await RunShapeInferPass(string.Empty, expand, weights);
        Assert.True(s is Const);
    }

    [Fact]
    public async Task TestFoldShapeOf()
    {
        var input = new Var("input", new TensorType(DataTypes.Int32, new Shape(1, 3, 240, 320)));
        var shape = ShapeOf(input);
        var shapePost = await RunShapeInferPass(string.Empty, shape);
        Assert.Equal(new long[] { 1, 3, 240, 320 }, ((TensorConst)shapePost).Value.ToArray<long>());
    }

    [Fact]
    public async Task TestExpandToRank()
    {
        var input = new Var("input", new TensorType(DataTypes.Int32, new Shape(1, 3, 240, 320)));
        var exp = Expand(1, Cast(Rank(input) - 0, DataTypes.Int64));
        var result = await RunShapeInferPass(string.Empty, exp);
        Assert.Equal(new[] { 1, 1, 1, 1 }, result.Evaluate().AsTensor().ToArray<int>());
    }
}

[AutoSetupTestMethod(InitSession = true)]
public class UnitTestDataFlowRewriteAndInferIntegrate : RewriteFixtrue
{
    public IAnalyzerManager AnalyzerMananger => CompileSession.GetRequiredService<IAnalyzerManager>();

    public T Dim1ExprToScalar<T>(Expr expr)
        where T : unmanaged, System.IEquatable<T>
        => ((TensorConst)expr).Value.Cast<T>()[0];

    [Fact]
    public async Task TestPaddingCompute()
    {
        var input = new Var("input", new TensorType(DataTypes.Int32, new Shape(1, 3, 33, 65)));
        var weights = Tensor.From<int>(Enumerable.Range(0, 3 * 3 * 3 * 16).ToArray(), new Shape(new[] { 16, 3, 3, 3 }));
        var (inH, inW) = Util.GetHW(input);
        var (fH, fW) = Util.GetHW(weights);
        using var exprPin1 = new ExprPinner(input, inH, inW, fH, fW);

        var inHPost = await RunShapeInferPass("inH", inH);
        var inWPost = await RunShapeInferPass("inW", inW);
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
        var paddingPost = await RunShapeInferPass("padding", padding, input);
        Assert.Equal(Tensor.From(new[] { 1, 1, 1, 1 }, new Shape(2, 2)), paddingPost);
    }

    [Fact]
    public async Task TestYolo20MinStructure()
    {
        var input = new Var("input", new TensorType(DataTypes.Int32, new Shape(new[] { 1, 240, 320, 3 })));
        var weights = Tensor.From<int>(Enumerable.Range(0, 3 * 3 * 3 * 16).ToArray(), new Shape(new[] { 16, 3, 3, 3 }));
        var bias = Tensor.From<int>(Enumerable.Range(0, 16).ToArray());
        var (inH, inW) = Util.GetHW(input);
        var (fH, fW) = Util.GetHW(weights);
        var strideH = 1;
        var strideW = 1;
        var dilationH = 1;
        var dilationW = 1;
        var padH = Util.GetWindowedPadding(inH, fH, strideH, dilationH, true);
        var padW = Util.GetWindowedPadding(inW, fW, strideW, dilationW, true);
        var stride = Tensor.From<int>(new[] { strideH, strideW }, new[] { 2 });
        var dilation = Tensor.From<int>(new[] { dilationH, dilationW }, new[] { 2 });
        var padding = Util.ConcatPadding(padH, padW);

        var conv = NN.Conv2D(
            NHWCToNCHW(input),
            NHWCToNCHW(weights),
            bias,
            stride,
            padding,
            dilation,
            PadMode.Constant,
            1);
        var convAfterTranspose = NCHWToNHWC(Clamp(conv, 0, 1));

        var postConvAfterTranspose = await RunShapeInferPass("convAfterTranspose", convAfterTranspose);
        Assert.True(CompilerServices.InferenceType(postConvAfterTranspose));
        Assert.Equal(new Shape(1, 240, 320, 16), postConvAfterTranspose.CheckedShape);

        var mul = Binary(BinaryOp.Mul, 1, convAfterTranspose);
        var max = Binary(BinaryOp.Max, convAfterTranspose, mul);

        // ReduceWindow2D
        var doubleV = Tensor.From<int>(new[] { 2, 2 }, new[] { 2 });
        var initValue = (Const)0;
        var (rInH, rInW) = Util.GetHW(max);
        var rPadH = Util.GetWindowedPadding(rInH, 2, 2, dilationH, true);
        var rPadW = Util.GetWindowedPadding(rInW, 2, 2, dilationW, true);
        var rPadding = Util.ConcatPadding(rPadH, rPadW);
        var reduce = NCHWToNHWC(ReduceWindow2D(ReduceOp.Max, NHWCToNCHW(max), initValue, doubleV, doubleV, rPadding, dilation, false, false));
        var post = await RunShapeInferPass("reduce", reduce);
        Assert.True(CompilerServices.InferenceType(post));
        Assert.Equal(new Shape(1, 120, 160, 16), post.CheckedShape);
    }

    [Fact]
    public async Task SliceForShapeIndex()
    {
        var input = new Var(new TensorType(DataTypes.Float32, new Shape(1, 7, 7, 75)));
        var slice = Util.ShapeIndex(input, 1);
        CompilerServices.InferenceType(slice);
        var post = await RunShapeInferPass("slice", slice);
        Assert.True(CompilerServices.InferenceType(post));
        Assert.True(post is Const);
        Assert.Equal(Shape.Scalar, post.CheckedShape);
    }

    [Fact]
    public async Task SoftMaxImporterProcess()
    {
        var input = new Var(new TensorType(DataTypes.Float32, new Shape(1, 3, 224, 224)));
        var axis = -1;
        var inShape = ShapeOf(input);
        using var inShapePin = new ExprPinner(inShape);
        Expr axisExprBefore = axis < 0
            ? axis + Rank(input)
            : Tensor.From<int>(new[] { axis });
        axisExprBefore.InferenceType();
        var axisExpr = await RunShapeInferPass("Axis", axisExprBefore, input);
        using var axisPin = new ExprPinner(axisExpr);
        Assert.Equal(3, ((TensorConst)axisExpr).Value.Cast<int>()[0]);

        var firstSliceBefore = Slice(inShape, new[] { 0 }, axisExpr, 1);
        firstSliceBefore.InferenceType();
        var firstSlice = await RunShapeInferPass("firstSlice", firstSliceBefore, input);
        using var firstSlicePin = new ExprPinner(firstSlice);
        Assert.Equal(new[] { 1, 3, 224 }, ((TensorConst)firstSlice).Value.ToArray<int>());

        var firstSizeBefore = Prod(firstSlice);
        firstSizeBefore.InferenceType();
        var firstSize = await RunShapeInferPass("firstSize", firstSizeBefore, input);
        Assert.Equal(1 * 3 * 224, ((TensorConst)firstSize).Value.ToScalar<int>());

        var secondBefore = Prod(Slice(inShape, axisExpr, Rank(input), 1));
        var secondSize = await RunShapeInferPass("secondSize", secondBefore, input);
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
        var v = Tensor.From<int>(new[] { 1, 2, 3 });
        var shape = Concat(
            new IR.Tuple(
                ShapeOf(v),
                new long[] { 1 },
                new long[] { 1 }),
            0);
        var afterShape = await RunShapeInferPass("Shape", shape);
        Assert.True(afterShape.InferenceType());
        Assert.Equal(new long[] { 3, 1, 1 }, afterShape);
        var b = Reshape(v, afterShape);
        b.InferenceType();
        Assert.Equal(new[] { 3, 1, 1 }, b.Evaluate().AsTensor().Dimensions.ToArray());

        var a = OnnxImporter.ReshapeToByChannel(v);
        var after = await RunShapeInferPass("ReshapeToByChannel", a);
        Assert.True(after.InferenceType());
        Assert.Equal(new[] { 3, 1, 1 }, after.Evaluate().AsTensor().Dimensions.ToArray());
    }

    [Fact]
    public void TestWithAnalysisInfoRewriteOnce()
    {
        var x = new Var(TensorType.Scalar(DataTypes.Int32));
        var y = new Var(TensorType.Scalar(DataTypes.Int32));
        var z = new Var(TensorType.Scalar(DataTypes.Int32));
        var m = new Var(TensorType.Scalar(DataTypes.Int32));
        var pre = new Function(m + (x + (z + (x + (y / y)))), new[] { x, y, z, m });
        CompilerServices.InferenceType(pre);

        var analysis = new Dictionary<Type, IAnalysisResult>
        {
            [typeof(IExprUserAnalysisResult)] = AnalyzerMananger.GetAnaylsis<IExprUserAnalysisResult>(pre),
        };

        var pass = new DataflowPass() { Name = "DataflowWithUsdByPass" };
        pass.Add<AnalysisReassociateAdd>();
        pass.Add<DivToConst>();
        var post = (Function)pass.RunAsync(pre, new() { AnalysisResults = analysis }).Result;

        Assert.True(post.Body is Call { Target: IR.Math.Binary { BinaryOp: BinaryOp.Sub }, Arguments: var param0 } && // m - (x + (z - (x + (1))))
                    param0[1] is Call { Target: IR.Math.Binary { BinaryOp: BinaryOp.Add }, Arguments: var param1 } && // x + (z - (x + (1)))
                    param1[1] is Call { Target: IR.Math.Binary { BinaryOp: BinaryOp.Sub }, Arguments: var param2 } && // z - (x + (1))
                    param2[1] is Call { Target: IR.Math.Binary { BinaryOp: BinaryOp.Add }, Arguments: var param3 } && // x + (1)
                    param3[1] is TensorConst);
    }

    [Fact]
    public void TestBroadcastNopPadOutputNames()
    {
        var input = new Var(new TensorType(DataTypes.Float32, new Shape(1, 3, 224, 224)));
        var pad = new Call(new Pad(PadMode.Constant), new Expr[] { input, new float[,] { { 0, 0 }, { 0, 0 }, { 0, 0 }, { 0, 0 } }, 0.0f });
        pad.Metadata.OutputNames = new string[] { "pad" };
        var pre = new Function(pad, new[] { input });
        var pass = new DataflowPass() { Name = "BroadcastNopPadOutputNamesUpPass" };
        pass.Add<Passes.Rules.Neutral.BroadcastNopPadOutputNames>();
        pass.Add<Passes.Rules.Neutral.FoldNopPad>();
        var post = (Function)pass.RunAsync(pre, new()).Result;
        Assert.True(post.Body.Metadata.OutputNames![0] == "pad");

        pad = new Call(new Pad(PadMode.Constant), new Expr[] { input, new float[,] { { 0, 0 }, { 0, 0 }, { 0, 0 }, { 0, 0 } }, 0.0f });
        input.Metadata.OutputNames = new string[] { "input" };
        pre = new Function(pad, new[] { input });
        pass = new DataflowPass() { Name = "BroadcastNopPadOutputNamesDownPass" };
        pass.Add<Passes.Rules.Neutral.BroadcastNopPadOutputNames>();
        post = (Function)pass.RunAsync(pre, new()).Result;
        Assert.True(post.Body is Call && post.Body.Metadata.OutputNames![0] == "input");
    }

    [Fact]
    public void TestBroadcastReshapeOutputNames()
    {
        var input = new Var(new TensorType(DataTypes.Float32, new Shape(1, 3, 224, 224)));
        var reshape = new Call(new Reshape(), new Expr[] { input, new int[] { 1, 224, 224, 3 } });
        reshape.Metadata.OutputNames = new string[] { "reshape" };
        var pre = new Function(reshape, new[] { input });
        var pass = new DataflowPass() { Name = "BroadcastReshapeOutputNamesUpPass" };
        pass.Add<Passes.Rules.Neutral.BroadcastReshapeOutputNames>();
        pass.Add<Passes.Rules.Neutral.FoldNopReshape>();
        var post = (Function)pass.RunAsync(pre, new()).Result;
        Assert.True(post.Body.Metadata.OutputNames![0] == "reshape");

        reshape = new Call(new Reshape(), new Expr[] { input, new int[] { 1, 224, 224, 3 } });
        input.Metadata.OutputNames = new string[] { "input" };
        pre = new Function(reshape, new[] { input });
        pass = new DataflowPass() { Name = "BroadcastReshapeOutputNamesDownPass" };
        pass.Add<Passes.Rules.Neutral.BroadcastReshapeOutputNames>();
        post = (Function)pass.RunAsync(pre, new()).Result;
        Assert.True(post.Body is Call && post.Body.Metadata.OutputNames![0] == "input");
    }

    [Fact]
    public void TestBroadcastTransposeOutputNames()
    {
        var input = new Var(new TensorType(DataTypes.Float32, new Shape(1, 3, 224, 224)));
        var transpose = new Call(new Transpose(), new Expr[] { input, new int[] { 0, 1, 2, 3 } });
        transpose.Metadata.OutputNames = new string[] { "transpose" };
        var pre = new Function(transpose, new[] { input });
        var pass = new DataflowPass() { Name = "BroadcastTransposeOutputNamesUpPass" };
        pass.Add<Passes.Rules.Neutral.BroadcastTransposeOutputNames>();
        pass.Add<Passes.Rules.Neutral.FoldNopTranspose>();
        var post = (Function)pass.RunAsync(pre, new()).Result;
        Assert.True(post.Body.Metadata.OutputNames![0] == "transpose");

        transpose = new Call(new Transpose(), new Expr[] { input, new int[] { 0, 1, 2, 3 } });
        input.Metadata.OutputNames = new string[] { "input" };
        pre = new Function(transpose, new[] { input });
        pass = new DataflowPass() { Name = "BroadcastTransposeOutputNamesDownPass" };
        pass.Add<Passes.Rules.Neutral.BroadcastTransposeOutputNames>();
        post = (Function)pass.RunAsync(pre, new()).Result;
        Assert.True(post.Body is Call && post.Body.Metadata.OutputNames![0] == "input");
    }

    private sealed class DivToConst : IRewriteRule
    {
        private static readonly Pattern SInputPattern = IsWildcard("x");

        /// <inheritdoc/>
        public IPattern Pattern { get; } = SInputPattern / SInputPattern;

        public Expr? GetReplace(IMatchResult result, RunPassContext options)
        {
            var x = (Expr)result["x"];
            return Tensor.FromScalar<int>(1).CastTo(x.CheckedDataType, CastMode.KDefault);
        }
    }

    private sealed class AnalysisReassociateAdd : IRewriteRule
    {
        /// <inheritdoc/>
        public IPattern Pattern { get; } = IsWildcard("x") + IsWildcard("y");

        public Expr? GetReplace(IMatchResult result, RunPassContext options)
        {
            var userAnalysis = options.GetAnalysis<IExprUserAnalysisResult>();
            var x = (Expr)result["x"];
            var y = (Expr)result["y"];
            if (userAnalysis[x].Count() == 1 && userAnalysis[y].Count() == 1)
            {
                return x - y;
            }

            return null;
        }
    }
}
