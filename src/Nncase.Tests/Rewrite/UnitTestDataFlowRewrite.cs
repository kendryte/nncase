﻿// Copyright (c) Canaan Inc. All rights reserved.
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
using Nncase.IR.Shapes;
using Nncase.IR.Tensors;
using Nncase.Passes;
using Nncase.Passes.Analysis;
using Nncase.Passes.Rules.Neutral;
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
    public UnitTestDataFlowRewrite()
    {
#if DEBUG
        CompileOptions.DumpFlags = Diagnostics.DumpFlags.Compile;
#endif
    }

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
        var pre = Concat(new IR.Tuple((Expr)lhs.ToTensor(), (Expr)rhs.ToTensor()), 1);
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
    //     var input = new Var("input", new TensorType(DataTypes.Int32, new RankedShape(new[] { 1, 3, 240, 320 })));
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
        var weights = new Var("weights", new TensorType(DataTypes.Float32, new RankedShape(1, 3, 224, 224)));
        _ = Util.ShapeIndex(weights, 0);
        var expand = Expand(0f, new RankedShape(Util.ShapeIndex(weights, 0)));
        var s = await RunShapeInferPass(string.Empty, expand, weights);
        Assert.True(s is Const);
    }

    [Fact]
    public void TestTileToExpand()
    {
        var input = new Var("input", new TensorType(DataTypes.Float32, new long[] { 1, 32, 256, 1 }));
        var tile = Tile(input, new long[] { 1, 1, 1, 2 });
        var expand = Expand(input, new long[] { 1, 32, 256, 2 });

        var input_tensor = IR.F.Random.Normal(DataTypes.Float32, 0, 1, 2, new long[] { 1, 32, 256, 1 }).Evaluate().AsTensor();
        var feedDict = new Dictionary<IVar, IValue>
        {
            { input, Value.FromTensor(input_tensor) },
        };

        var pre = new Function(tile, new[] { input });
        var pass = new DataflowPass() { Name = "TileToExpand" };
        pass.Add<Passes.Rules.Neutral.TileToExpand>();

        var post = (Function)pass.RunAsync(pre, new()).Result;
        Assert.Equal(expand.Evaluate(feedDict).AsTensor().ToArray<float>(), post.Body.Evaluate(feedDict).AsTensor().ToArray<float>());
    }

    [Fact]
    public async Task TestFoldShapeOf()
    {
        var input = new Var("input", new TensorType(DataTypes.Int32, new RankedShape(1, 3, 240, 320)));
        var shape = ShapeOf(input);
        var shapePost = await RunShapeInferPass(string.Empty, shape);
        Assert.Equal(new long[] { 1, 3, 240, 320 }, ((TensorConst)shapePost).Value.ToArray<long>());
    }

    [Fact]
    public async Task TestExpandToRank()
    {
        var input = new Var("input", new TensorType(DataTypes.Int32, new RankedShape(1, 3, 240, 320)));
        var exp = Expand(1, new RankedShape(Rank(input).AsDim() - 0L));
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
        var input = new Var("input", new TensorType(DataTypes.Int32, new RankedShape(1, 3, 33, 65)));
        var weights = Tensor.From<int>(Enumerable.Range(0, 3 * 3 * 3 * 16).ToArray(), new RankedShape(new[] { 16, 3, 3, 3 }));
        var (inH, inW) = Util.GetHW(input);
        var (fH, fW) = Util.GetHW(weights);
        using var exprPin1 = new ExprPinner(input, inH, inW, fH, fW);

        var inHPost = await RunShapeInferPass("inH", inH);
        var inWPost = await RunShapeInferPass("inW", inW);
        Assert.Equal(33, ((DimConst)inHPost).Value);
        Assert.Equal(65, ((DimConst)inWPost).Value);
        var strideH = 1;
        var strideW = 1;
        var dilationH = 1;
        var dilationW = 1;
        var padH = TypeInference.GetWindowedPadding(inH, fH, strideH, dilationH, true);
        var padW = TypeInference.GetWindowedPadding(inW, fW, strideW, dilationW, true);
        var padding = new Paddings(padH, padW);

        // Assert.True(CompilerServices.InferenceType(padding));
        var paddingPost = (Paddings)await RunShapeInferPass("padding", padding, input);
        Assert.Equal(new long[,] { { 1, 1 }, { 1, 1 } }, paddingPost.ToValueArray());
    }

    [Fact]
    public async Task TestYolo20MinStructure()
    {
        var input = new Var("input", new TensorType(DataTypes.Int32, new RankedShape(new[] { 1, 240, 320, 3 })));
        var weights = Tensor.From<int>(Enumerable.Range(0, 3 * 3 * 3 * 16).ToArray(), new RankedShape(new[] { 16, 3, 3, 3 }));
        var bias = Tensor.From<int>(Enumerable.Range(0, 16).ToArray());
        var (inH, inW) = Util.GetHW(input);
        var (fH, fW) = Util.GetHW(weights);
        var strideH = 1;
        var strideW = 1;
        var dilationH = 1;
        var dilationW = 1;
        var padH = TypeInference.GetWindowedPadding(inH, fH, strideH, dilationH, true);
        var padW = TypeInference.GetWindowedPadding(inW, fW, strideW, dilationW, true);
        var stride = Tensor.From<int>(new[] { strideH, strideW }, [2]);
        var dilation = new[] { dilationH, dilationW };
        var padding = new Paddings(padH, padW);

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
        Assert.Equal(new RankedShape(1, 240, 320, 16), postConvAfterTranspose.CheckedShape);

        var mul = Binary(BinaryOp.Mul, 1, convAfterTranspose);
        var max = Binary(BinaryOp.Max, convAfterTranspose, mul);

        // ReduceWindow2D
        var doubleV = new[] { 2, 2 };
        var initValue = (Const)0;
        var (rInH, rInW) = Util.GetHW(max);
        var rPadH = TypeInference.GetWindowedPadding(rInH, 2, 2, dilationH, true);
        var rPadW = TypeInference.GetWindowedPadding(rInW, 2, 2, dilationW, true);
        var rPadding = new Paddings(rPadH, rPadW);
        var reduce = NCHWToNHWC(ReduceWindow2D(ReduceOp.Max, NHWCToNCHW(max), initValue, doubleV, doubleV, rPadding, dilation, false, false));
        var post = await RunShapeInferPass("reduce", reduce);
        Assert.True(CompilerServices.InferenceType(post));
        Assert.Equal(new RankedShape(1, 120, 160, 16), post.CheckedShape);
    }

    [Fact]
    public async Task SliceForShapeIndex()
    {
        var input = new Var(new TensorType(DataTypes.Float32, new RankedShape(1, 7, 7, 75)));
        var slice = Util.ShapeIndex(input, 1);
        CompilerServices.InferenceType(slice);
        var post = await RunShapeInferPass("slice", slice);
        Assert.True(CompilerServices.InferenceType(post));
        Assert.True(post is DimConst);
        Assert.Equal(new DimensionType(DimensionKind.Fixed), post.CheckedType);
    }

    [Fact]
    public async Task SoftMaxImporterProcess()
    {
        var input = new Var(new TensorType(DataTypes.Float32, new RankedShape(1, 3, 224, 224)));
        var axis = -1L;
        var inShape = (RankedShape)ShapeOf(input).AsShape();
        var axisExprBefore = new RankedShape(axis < 0
            ? axis + Rank(input).AsDim()
            : axis);
        axisExprBefore.InferenceType();
        var axisExpr = (Shape)await RunShapeInferPass("Axis", axisExprBefore, input);
        Assert.Equal(3, axisExpr[0].FixedValue);

        var firstSliceBefore = new RankedShape(inShape[0..(int)axisExpr[0].FixedValue]);
        firstSliceBefore.InferenceType();
        var firstSlice = (RankedShape)await RunShapeInferPass("firstSlice", firstSliceBefore, input);
        using var firstSlicePin = new ExprPinner(firstSlice);
        Assert.Equal([1, 3, 224], firstSlice.ToValueArray());

        var firstSizeBefore = firstSlice.Prod();
        firstSizeBefore.InferenceType();
        var firstSize = (Dimension)await RunShapeInferPass("firstSize", firstSizeBefore, input);
        Assert.Equal(1 * 3 * 224, firstSize.FixedValue);

        var secondBefore = new RankedShape(inShape[(int)axisExpr[0].FixedValue..]).Prod();
        var secondSize = (Dimension)await RunShapeInferPass("secondSize", secondBefore, input);
        Assert.Equal(224, secondSize.FixedValue);

        var beforeShape = new RankedShape(firstSize, secondSize);
        var afterShape = ShapeOf(input).AsShape();
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
        var shape = new RankedShape([.. ShapeOf(v).AsShape(), 1, 1]);
        var afterShape = (Shape)await RunShapeInferPass("Shape", shape);
        Assert.True(afterShape.InferenceType());
        Assert.Equal(new long[] { 3, 1, 1 }, afterShape);
        var b = Reshape(v, afterShape);
        b.InferenceType();
        Assert.Equal([3, 1, 1], b.Evaluate().AsTensor().Dimensions.ToArray());

        var a = OnnxImporter.ReshapeToByChannel(v);
        var after = await RunShapeInferPass("ReshapeToByChannel", a);
        Assert.True(after.InferenceType());
        Assert.Equal([3, 1, 1], after.Evaluate().AsTensor().Dimensions.ToArray());
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

        var analysis = new Dictionary<System.Type, IAnalysisResult>
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

    [Theory]
    [InlineData([true, 0])]
    [InlineData([false, 1])]
    public void TestPaperCase(bool left, int count)
    {
        var atype = new TensorType(DataTypes.Float32, new[] { 30, 40, 20 });
        var a = new Var(atype);
        var btype = new TensorType(DataTypes.Float32, new[] { 30, 20, 40 });
        var b = new Var(btype);
        Function pre;
        {
            // A: [2, 0, 1],  invA: [1,2,0]
            // B: [1, 0, 2],  invB: [1,0,2]
            var transA = IR.F.Tensors.Transpose(a, new[] { 2, 0, 1 }); // 20,30,40;
            var transB = IR.F.Tensors.Transpose(b, new[] { 1, 0, 2 }); // 20,30,40;
            var exp = IR.F.Math.Cos(transA + transB); // 20,30,40;
            var transC = IR.F.Tensors.Transpose(exp, new[] { 1, 2, 0 }); // 30,40,20
            pre = new IR.Function(transC, a, b);
        }

        using var scope = new Diagnostics.DumpScope(count.ToString(), Diagnostics.DumpFlags.Rewrite | Diagnostics.DumpFlags.EGraphCost);
#if DEBUG
        Diagnostics.DumpScope.Current.DumpIR(pre, $"pre");
#endif
        BaseExpr post = pre;
        if (left)
        {
            post = new DataFlowRewriter(new Passes.Rules.Neutral.CombineBinaryLeftTranspose(), new()).Rewrite(post);
        }
        else
        {
            post = new DataFlowRewriter(new Passes.Rules.Neutral.CombineBinaryRightTranspose(), new()).Rewrite(post);
        }
#if DEBUG
        var name = left ? "Left" : "Rigth";
        Diagnostics.DumpScope.Current.DumpIR(post, $"CombineBinary{name}Transpose");
#endif

        post = new DataFlowRewriter(new Passes.Rules.Neutral.CombineUnaryTranspose(), new()).Rewrite(post);
#if DEBUG
        Diagnostics.DumpScope.Current.DumpIR(post, $"CombineUnaryTranspose");
#endif
        post = new DataFlowRewriter(new Passes.Rules.Neutral.FoldTwoTransposes(), new()).Rewrite(post);
#if DEBUG
        Diagnostics.DumpScope.Current.DumpIR(post, $"FoldTwoTransposes");
#endif
        post = new DataFlowRewriter(new Passes.Rules.Neutral.FoldNopTranspose(), new()).Rewrite(post);
#if DEBUG
        Diagnostics.DumpScope.Current.DumpIR(post, $"FoldNopTranspose");
#endif

#if DEBUG
        Diagnostics.DumpScope.Current.DumpIR(post, "post");
#endif

        var feedDict = new Dictionary<IVar, IValue>()
        {
            { a, IR.F.Random.Normal(atype.DType, 0, 1, 2, atype.Shape.ToValueArray()).Evaluate() },
            { b, IR.F.Random.Normal(btype.DType, 0, 1, 2, btype.Shape.ToValueArray()).Evaluate() },
        };

        var preValue = pre.Body.Evaluate(feedDict);
        var postValue = ((Function)post).Body.Evaluate(feedDict);
        Assert.True(Comparator.Compare(preValue, postValue));
    }

    [Fact]
    public void TestBroadcastNopPadOutputNames()
    {
        var input = new Var(new TensorType(DataTypes.Float32, new RankedShape(1, 3, 224, 224)));
        var pad = IR.F.NN.Pad(input, new int[,] { { 0, 0 }, { 0, 0 }, { 0, 0 }, { 0, 0 } }, PadMode.Constant, 0.0f);
        pad.Metadata.OutputNames = new string[] { "pad" };
        var pre = new Function(pad, new[] { input });
        var pass = new DataflowPass() { Name = "BroadcastNopPadOutputNamesUpPass" };
        pass.Add<Passes.Rules.Neutral.BroadcastNopPadOutputNames>();
        pass.Add<Passes.Rules.Neutral.FoldNopPad>();
        var post = (Function)pass.RunAsync(pre, new()).Result;
        Assert.True(post.Body.Metadata.OutputNames![0] == "pad");

        pad = IR.F.NN.Pad(input, new int[,] { { 0, 0 }, { 0, 0 }, { 0, 0 }, { 0, 0 } }, PadMode.Constant, 0.0f);
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
        var input = new Var(new TensorType(DataTypes.Float32, new RankedShape(1, 3, 224, 224)));
        var reshape = IR.F.Tensors.Reshape(input, new int[] { 1, 224, 224, 3 });
        reshape.Metadata.OutputNames = new string[] { "reshape" };
        var pre = new Function(reshape, new[] { input });
        var pass = new DataflowPass() { Name = "BroadcastReshapeOutputNamesUpPass" };
        pass.Add<Passes.Rules.Neutral.BroadcastReshapeOutputNames>();
        pass.Add<Passes.Rules.Neutral.FoldNopReshape>();
        var post = (Function)pass.RunAsync(pre, new()).Result;
        Assert.True(post.Body.Metadata.OutputNames![0] == "reshape");

        reshape = IR.F.Tensors.Reshape(input, new int[] { 1, 224, 224, 3 });
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
        var input = new Var(new TensorType(DataTypes.Float32, new RankedShape(1, 3, 224, 224)));
        var transpose = IR.F.Tensors.Transpose(input, new int[] { 0, 1, 2, 3 });
        transpose.Metadata.OutputNames = new string[] { "transpose" };
        var pre = new Function(transpose, new[] { input });
        var pass = new DataflowPass() { Name = "BroadcastTransposeOutputNamesUpPass" };
        pass.Add<Passes.Rules.Neutral.BroadcastTransposeOutputNames>();
        pass.Add<Passes.Rules.Neutral.FoldNopTranspose>();
        var post = (Function)pass.RunAsync(pre, new()).Result;
        Assert.True(post.Body.Metadata.OutputNames![0] == "transpose");

        transpose = IR.F.Tensors.Transpose(input, new int[] { 0, 1, 2, 3 });
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

        public BaseExpr? GetReplace(IMatchResult result, RunPassContext options)
        {
            var x = (Expr)result["x"];
            return (Expr)Tensor.FromScalar<int>(1).CastTo(x.CheckedDataType, CastMode.KDefault);
        }
    }

    private sealed class AnalysisReassociateAdd : IRewriteRule
    {
        /// <inheritdoc/>
        public IPattern Pattern { get; } = IsWildcard("x") + IsWildcard("y");

        public BaseExpr? GetReplace(IMatchResult result, RunPassContext options)
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
