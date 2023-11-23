// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using Nncase.Diagnostics;
using Nncase.IR;
using Nncase.IR.Tensors;
using Nncase.Passes;
using Nncase.Passes.Rules.Neutral;
using Nncase.Passes.Rules.ShapeBucket;
using Nncase.Quantization;
using Nncase.Tests.ReWrite.FusionTest;
using Nncase.Tests.TestFixture;
using Nncase.Tests.TransformTest;
using Nncase.Utilities;
using Xunit;
using Xunit.Abstractions;
using static Nncase.IR.F.Math;
using static Nncase.IR.F.NN;
using static Nncase.IR.F.Tensors;
using static Nncase.Tests.Rules.ShapeBucket.ShapeBucketTestHelper;
using Reshape = Nncase.IR.Tensors.Reshape;

namespace Nncase.Tests.Rules.ShapeBucket;

[AutoSetupTestMethod(InitSession = true)]
public class ShapeBucketTest : TransformTestBase
{
    [Fact]
    public void TestFindVar()
    {
        var v1 = new Var(new TensorType(DataTypes.Int32, Shape.Scalar));
        var v2 = new Var(new TensorType(DataTypes.Int32, Shape.Scalar));
        var expr = ((v1 * 4) + (v2 * 3)) * 2;
        var visitor = new FindVar();
        visitor.Visit(expr);
        Assert.Equal(visitor.Vars, new HashSet<Var>(new[] { v1, v2 }));
    }

    [Fact]
    public void TestBucketPad()
    {
        var input = Testing.Rand<float>(1, 2, 16, 16);
        var fixedShape = new[] { 1, 3, 24, 24 };
        var p = new Call(new BucketPad(), input, fixedShape);
        var (_, kmodel) = Testing.BuildKModel("test", new IRModule(new Function(p)), CompileSession);
        var result = Testing.RunKModel(kmodel, "call_arg", Array.Empty<Tensor>());
        var pads = fixedShape - Cast(ShapeOf(input), DataTypes.Int32);
        var paddings = Transpose(
            Stack(new IR.Tuple(Enumerable.Repeat(0, fixedShape.Length).ToArray(), pads), 0),
            new[] { 1, 0 });
        var fixedInput = IR.F.NN.Pad(input, paddings, PadMode.Constant, Cast(0, input.ElementType));
        var fixedResult = new Call(new FixShape(), fixedInput, fixedShape);
        var origin = fixedResult.Evaluate();
        var cos = Comparator.CosSimilarity(origin, result)[0];
        Assert.True(cos > 0.999);
    }

    [Fact]
    public async Task TestSingleVarFusionBucket()
    {
        var mainVar = new Var(new TensorType(DataTypes.Float32, new[] { 1, Dimension.Unknown, 24, 24 }));
        var dimVar = Scalar("dimVar");
        CompileOptions.ShapeBucketOptions.Enable = true;
        CompileOptions.ShapeBucketOptions.SegmentsCount = 2;
        CompileOptions.ShapeBucketOptions.RangeInfo =
            new Dictionary<string, (int Min, int Max)> { { "dimVar", (1, 20) } };
        CompileOptions.ShapeBucketOptions.VarMap = new Dictionary<Var, Expr[]> { { mainVar, new Expr[] { 1, dimVar, 24, 24 } } };

        var input = Testing.Rand<float>(1, 3, 24, 24);
        var fusionVar = new Var(new TensorType(DataTypes.Float32, new[] { 1, Dimension.Unknown, 24, 24 }));
        var f = new BucketFusion("MatMul_0", "stackvm", IR.F.Math.MatMul(fusionVar, fusionVar), new[] { fusionVar }, new[] { dimVar });
        var main = new Function("main", new Call(f, mainVar), mainVar);
        var shape = new Dictionary<BucketFusion, FusionShapeData[]>();
        await new RecordFusionShape(shape).RunAsync(main, new());
        TestMatchedCore(
            main.Body!,
            new Dictionary<Var, IValue> { { mainVar, Value.FromTensor(input) } },
            new FusionBucket(shape));
    }

    [Fact]
    public async Task TestRebuild()
    {
        var mainVar = new Var(new TensorType(DataTypes.Float32, new[] { 1, Dimension.Unknown, 24, 24 }));
        var dimVar = Scalar("dimVar");
        CompileOptions.ShapeBucketOptions.Enable = true;
        CompileOptions.ShapeBucketOptions.SegmentsCount = 2;
        CompileOptions.ShapeBucketOptions.RangeInfo =
            new Dictionary<string, (int Min, int Max)> { { "dimVar", (1, 20) } };
        CompileOptions.ShapeBucketOptions.VarMap = new Dictionary<Var, Expr[]> { { mainVar, new Expr[] { 1, dimVar, 24, 24 } } };

        var input = Testing.Rand<float>(1, 3, 24, 24);
        var fusionVar = new Var(new TensorType(DataTypes.Float32, new[] { 1, Dimension.Unknown, 24, 24 }));
        var shapeVar = new Var(new TensorType(DataTypes.Int64, new[] { 4 }));
        var body = IR.F.Math.MatMul(Reshape(fusionVar, shapeVar), fusionVar);
        var f = new BucketFusion("MatMul_0", "stackvm", body, new[] { fusionVar, shapeVar }, new[] { dimVar });
        var main = new Function("main", new Call(f, mainVar, Stack(new IR.Tuple(new[] { 1L, ShapeOf(mainVar)[1], 24L, 24L }), 0)));
        var shape = new Dictionary<BucketFusion, FusionShapeData[]>();
        await new RecordFusionShape(shape).RunAsync(main, new());
        var newBody = TestMatchedCore(
            main.Body!,
            new Dictionary<Var, IValue> { { mainVar, Value.FromTensor(input) } },
            new RebuildBucket(shape));
        Assert.True(newBody is Call { Target: IR.Math.MatMul });
    }

    [Fact]
    public async Task TestTupleOutput()
    {
        var mainVar = new Var(new TensorType(DataTypes.Float32, new[] { 1, Dimension.Unknown, 24, 24 }));
        var dimVar = Scalar("dimVar");
        CompileOptions.ShapeBucketOptions.Enable = true;
        CompileOptions.ShapeBucketOptions.SegmentsCount = 2;
        CompileOptions.ShapeBucketOptions.RangeInfo =
            new Dictionary<string, (int Min, int Max)> { { "dimVar", (1, 20) } };
        CompileOptions.ShapeBucketOptions.VarMap = new Dictionary<Var, Expr[]> { { mainVar, new Expr[] { 1, dimVar, 24, 24 } } };

        var input = Testing.Rand<float>(1, 3, 24, 24);
        var fusionVar = new Var(new TensorType(DataTypes.Float32, new[] { 1, Dimension.Unknown, 24, 24 }));
        var mm = IR.F.Math.MatMul(fusionVar, fusionVar);
        var body = new IR.Tuple(mm, mm);
        var f = new BucketFusion("MatMul_0", "stackvm", body, new[] { fusionVar }, new[] { dimVar });
        var main = new Function("main", new Call(f, mainVar), mainVar);
        var shape = new Dictionary<BucketFusion, FusionShapeData[]>();
        await new RecordFusionShape(shape).RunAsync(main, new());
        TestMatchedCore(
            main.Body!,
            new Dictionary<Var, IValue> { { mainVar, Value.FromTensor(input) } },
            new FusionBucket(shape));
    }

    [Fact]
    public async Task TestDoubleVarFusionBucket()
    {
        var mainVarLhs = new Var(new TensorType(DataTypes.Float32, new[] { 1, Dimension.Unknown, 24, 24 }));
        var mainVarRhs = new Var(new TensorType(DataTypes.Float32, new[] { 1, Dimension.Unknown, 24, 24 }));
        var dimVar1 = Scalar("dimVar1");
        var dimVar2 = Scalar("dimVar2");
        CompileOptions.ShapeBucketOptions.Enable = true;
        CompileOptions.ShapeBucketOptions.SegmentsCount = 5;
        CompileOptions.ShapeBucketOptions.RangeInfo =
            new Dictionary<string, (int Min, int Max)>
            {
                { "dimVar1", (1, 20) },
                { "dimVar2", (1, 20) },
            };
        CompileOptions.ShapeBucketOptions.VarMap = new Dictionary<Var, Expr[]>
        {
            { mainVarLhs, new Expr[] { 1, dimVar1, 24, 24 } },
            { mainVarRhs, new Expr[] { 1, dimVar2, 24, 24 } },
        };

        var inputLhs = Testing.Rand<float>(1, 3, 24, 24);
        var inputRhs = Testing.Rand<float>(1, 3, 24, 24);
        var fusionVar = new Var(new TensorType(DataTypes.Float32, new[] { 1, Dimension.Unknown, 24, 24 }));
        var f = new BucketFusion("MatMul_0", "stackvm", IR.F.Math.MatMul(fusionVar, fusionVar), new[] { fusionVar }, new[] { dimVar1 });
        var main = new Function("main", new Call(f, mainVarLhs, mainVarRhs), mainVarLhs, mainVarRhs);
        var shape = new Dictionary<BucketFusion, FusionShapeData[]>();
        await new RecordFusionShape(shape).RunAsync(main, new());
        TestMatchedCore(
            main.Body!,
            new Dictionary<Var, IValue>
            {
                { mainVarLhs, Value.FromTensor(inputLhs) },
                { mainVarRhs, Value.FromTensor(inputRhs) },
            },
            new FusionBucket(shape));
    }

    [Fact]
    public async Task TestDoubleVarWithMultiDimEffect()
    {
        var mainVarLhs = new Var(new TensorType(DataTypes.Float32, new[] { 1, Dimension.Unknown, 24, 24 }));
        var mainVarRhs = new Var(new TensorType(DataTypes.Float32, new[] { Dimension.Unknown, 1, 24, 24 }));
        var dimVar1 = Scalar("dimVar1");
        var dimVar2 = Scalar("dimVar2");
        CompileOptions.ShapeBucketOptions.Enable = true;
        CompileOptions.ShapeBucketOptions.SegmentsCount = 5;
        CompileOptions.ShapeBucketOptions.RangeInfo =
            new Dictionary<string, (int Min, int Max)>
            {
                { "dimVar1", (1, 20) },
                { "dimVar2", (1, 20) },
            };
        CompileOptions.ShapeBucketOptions.VarMap = new Dictionary<Var, Expr[]>
        {
            { mainVarLhs, new Expr[] { 1, dimVar1, 24, 24 } },
            { mainVarRhs, new Expr[] { dimVar2, 1, 24, 24 } },
        };

        var inputLhs = Testing.Rand<float>(1, 3, 24, 24);
        var inputRhs = Testing.Rand<float>(3, 1, 24, 24);
        var fusionVar = new Var(new TensorType(DataTypes.Float32, new[] { 1, Dimension.Unknown, 24, 24 }));
        var f = new BucketFusion("MatMul_0", "stackvm", IR.F.Math.MatMul(fusionVar, fusionVar), new[] { fusionVar }, new[] { dimVar1 });
        var main = new Function("main", new Call(f, mainVarLhs, mainVarRhs), mainVarLhs, mainVarRhs);
        var shape = new Dictionary<BucketFusion, FusionShapeData[]>();
        await new RecordFusionShape(shape).RunAsync(main, new());
        TestMatchedCore(
            main.Body!,
            new Dictionary<Var, IValue>
            {
                { mainVarLhs, Value.FromTensor(inputLhs) },
                { mainVarRhs, Value.FromTensor(inputRhs) },
            },
            new FusionBucket(shape));
    }

    private Var Scalar(string name) => new Var(name, new TensorType(DataTypes.Int32, Shape.Scalar));
}

[AutoSetupTestMethod(InitSession = true)]
public class TestMergePrevCallToFusion : TransformTestBase
{
    [Fact]
    public void TestMergePrevCallSingleInput()
    {
        var input = Testing.Rand<float>(1, 3, 24, 24);
        var inputVar = MakeVar(input);
        var transpose = Transpose(inputVar, new[] { 3, 2, 1, 0 });
        var v = new Var(transpose.CheckedType);
        var abs = Abs(v);
        var f = new BucketFusion("stackvm", abs, new[] { v }, Array.Empty<Var>());
        var c = new Call(f, transpose);
        TestMatched<MergePrevCallToFusion>(c, new Dictionary<Var, IValue> { { inputVar, Value.FromTensor(input) } });
    }

    [Fact]
    public void TestBodyMultiInputMergeLeft()
    {
        var mainInput0 = Testing.Rand<float>(1, 3, 24, 24);
        var mainInput1 = Testing.Rand<float>(1, 3, 24, 24);
        var mainVar0 = MakeVar(mainInput0);
        var mainVar1 = MakeVar(mainInput0);
        var fusionVar0 = MakeVar(mainInput0);
        var fusionVar1 = MakeVar(mainInput1);
        var concat = Concat(new IR.Tuple(fusionVar0, fusionVar1), 0);
        var f = new BucketFusion("stackvm", concat, new[] { fusionVar0, fusionVar1 }, Array.Empty<Var>());
        var abs = Abs(mainVar0);
        var c = new Call(f, abs, mainVar1);
        TestMatched<MergePrevCallToFusion>(
            c,
            new Dictionary<Var, IValue>
            {
                { mainVar0, Value.FromTensor(mainInput0) }, { mainVar1, Value.FromTensor(mainInput1) },
            });
    }

    [Fact]
    public void TestBodyMultiInputMergeRight()
    {
        var mainInput0 = Testing.Rand<float>(1, 3, 24, 24);
        var mainInput1 = Testing.Rand<float>(1, 3, 24, 24);
        var mainVar0 = MakeVar(mainInput0);
        var mainVar1 = MakeVar(mainInput0);
        var fusionVar0 = MakeVar(mainInput0);
        var fusionVar1 = MakeVar(mainInput1);
        var concat = Concat(new IR.Tuple(fusionVar0, fusionVar1), 0);
        var f = new BucketFusion("stackvm", concat, new[] { fusionVar0, fusionVar1 }, Array.Empty<Var>());
        var abs = Abs(mainVar1);
        var c = new Call(f, Softmax(mainInput0, 0), abs);
        TestMatched<MergePrevCallToFusion>(
            c,
            new Dictionary<Var, IValue>
            {
                { mainVar0, Value.FromTensor(mainInput0) }, { mainVar1, Value.FromTensor(mainInput1) },
            });
    }

    [Fact]
    public void TestPrevMultiInputForDynamicReshape()
    {
        // fusion
        var fusionVar = new Var(new TensorType(DataTypes.Float32, new[] { 1, 3, 24, 24 }));
        var transpose = Transpose(fusionVar, new[] { 3, 2, 1, 0 });
        var f = new BucketFusion("stackvm", transpose, new[] { fusionVar }, Array.Empty<Var>());

        // input
        var input = Testing.Rand<float>(3, 24, 24);
        var inputVar = new Var(new TensorType(input.ElementType, input.Shape));
        var newShape = Concat(new IR.Tuple(new[] { 1L }, ShapeOf(inputVar)), 0);
        var reshape = Reshape(Abs(inputVar), newShape);
        var c = new Call(f, reshape);
        TestMatched<MergePrevCallToFusion>(c, new Dictionary<Var, IValue> { { inputVar, Value.FromTensor(input) } });
    }

    [Fact]
    public void TestForMergeConcat()
    {
        var input0 = Testing.Rand<float>(1, 3, 24, 24);
        var input1 = Testing.Rand<float>(1, 3, 24, 24);
        var inputVar0 = MakeVar(input0);
        var inputVar1 = MakeVar(input1);
        var concat = Concat(new IR.Tuple(inputVar0, inputVar1), 0);
        var v = new Var(concat.CheckedType);
        var abs = Abs(v);
        var f = new BucketFusion("stackvm", abs, new[] { v }, Array.Empty<Var>());
        var c = new Call(f, concat);
        TestMatched<MergePrevCallToFusion>(
            c,
            new Dictionary<Var, IValue>
            {
                { inputVar0, Value.FromTensor(input0) }, { inputVar1, Value.FromTensor(input1) },
            });
    }

    [Fact]
    public void TestMatMulAndConcat()
    {
        var lhs = new Var("i0", new TensorType(DataTypes.Float32, new[] { 1, 3, 24, 24 }));
        var rhs = new Var("i1", new TensorType(DataTypes.Float32, new[] { 2, 3, 24, 24 }));
        var mm = IR.F.Math.MatMul(lhs, rhs);
        var f = new BucketFusion("stackvm", mm, new[] { lhs, rhs }, Array.Empty<Var>());

        var input0 = Testing.Rand<float>(1, 3, 24, 24);
        var input1 = Testing.Rand<float>(1, 3, 24, 24);
        var input2 = Testing.Rand<float>(1, 3, 24, 24);
        var inputVar0 = MakeVar(input0);
        var inputVar1 = MakeVar(input1);
        var inputVar2 = MakeVar(input2);
        var concat = Concat(new IR.Tuple(inputVar1, inputVar2), 0);
        var c = new Call(f, Softmax(inputVar0, 0), concat);
        TestMatched<MergePrevCallToFusion>(
            c,
            new Dictionary<Var, IValue>
            {
                { inputVar0, Value.FromTensor(input0) },
                { inputVar1, Value.FromTensor(input1) },
                { inputVar2, Value.FromTensor(input2) },
            });
    }

    [Fact]
    public void TestAfterMergeSameInput()
    {
        var input = Testing.Rand<float>(1, 3, 24, 24);
        var inputVar = new Var(new TensorType(input.ElementType, input.Shape));
        var abs = Abs(inputVar);

        var fusionVar0 = new Var(new TensorType(input.ElementType, input.Shape));
        var fusionVar1 = new Var(new TensorType(input.ElementType, input.Shape));
        var concat = Concat(new IR.Tuple(fusionVar0, fusionVar1), 0);
        var f = new BucketFusion("stackvm", concat, new Var[] { fusionVar0, fusionVar1 }, Array.Empty<Var>());
        var c = new Call(f, Sqrt(abs), Neg(abs));
        TestMatched<MergePrevCallToFusion>(c, new Dictionary<Var, IValue> { { inputVar, Value.FromTensor(input) } });
    }

    [Fact]
    public void TestMatMulReshape()
    {
        // 左边的表达式是右边表达式的一部分
        // 重新构造prev的call,使用新的var来替换，因此在替换%0的时候，ShapeOf()的参数也变成了var，但实际上ShapeOf的参数应当还是原始的
        // %0 = Add(BinaryOp.Add, %var_88: f32[1,3,24,24], const(f32[1] : {1f})) 2 -1378211376: // f32[1,3,24,24]
        // %1 = ShapeOf(%0) 1 2067123334: // i64[4]
        // %2 = Gather(%1, const(i64 : 0), const(i64 : 0)) 1 805902410: // i64
        // %3 = Reshape(%2, const(i64[1] : {1L})) 1 -466610003: // i64[1]
        // %4 = (%3, const(i64[1] : {3L}), const(i64[1] : {24L}), const(i64[1] : {24L})): // (i64[1], i64[1], i64[1], i64[1])
        //
        // %5 = Concat(%4, const(i32 : 0)) 1 80776753: // i64[4]
        // %6 = Reshape(%0, %5) 2 -1638748643: // f32[?,?,?,?]
        // %7 = (%6): // (f32[?,?,?,?])
        var input = Testing.Rand<float>(1, 3, 24, 24);
        var lhs = MakeVar(input);
        var add = Add(lhs, new[] { 1f });
        var rhs = Reshape(add, Concat(
            new IR.Tuple(Reshape(Gather(ShapeOf(add), 0, 0L), new[] { 1L }), new[] { 3L }, new[] { 24L }, new[] { 24L }), 0));

        var lhsVar = new Var("lhs", new TensorType(input.ElementType, input.Shape));
        var rhsVar = new Var("rhs", new TensorType(input.ElementType, input.Shape));
        var mm = IR.F.Math.MatMul(lhsVar, rhsVar);
        var f = new BucketFusion("stackvm", mm, new Var[] { lhsVar, rhsVar }, Array.Empty<Var>());
        var c = new Call(f, lhs, rhs);
        TestMatched<MergePrevCallToFusion>(c, new Dictionary<Var, IValue> { { lhs, Value.FromTensor(input) } });
    }

    [Fact]
    public void TestMergeStackWithConstant()
    {
        var input0 = Testing.Rand<float>(1, 3, 24, 24);
        var input1 = Testing.Rand<float>(1, 3, 24, 24);
        var lhs = MakeVar(input0);
        var rhs = MakeVar(input1);

        var scalarInput = Testing.Rand<float>();
        var scalarInputVar = MakeVar(scalarInput);
        var other = Stack(new IR.Tuple(scalarInputVar, 1f, 2f), 0);

        var lhsVar = MakeVar(input0);
        var rhsVar = MakeVar(input1);
        var otherVar = MakeVar(other);
        var mm = IR.F.Math.MatMul(lhsVar, rhsVar);
        var f = new BucketFusion("stackvm", mm * otherVar[1], new Var[] { lhsVar, rhsVar, otherVar }, Array.Empty<Var>());
        var c = new Call(f, lhs, rhs, other);
        var result = TestMatched<MergePrevCallToFusion>(c, new Dictionary<Var, IValue>
        {
            { lhs, Value.FromTensor(input0) },
            { rhs, Value.FromTensor(input1) },
            { scalarInputVar, Value.FromTensor(scalarInput) },
        });
        var fusion = GetResultFusion(result);

        // constant should not be var
        Assert.Equal(3, fusion.Parameters.Length);
    }

    // v1:add(a1, an) v2:add(a2, an)
    // fusion(v1, v2)
    //    |
    // fusion(a1, a2, an)
    [Fact]
    public void TestSameInputMerge()
    {
        var input0 = Testing.Rand<float>(1, 3, 24, 24);
        var input1 = Testing.Rand<float>(1, 3, 24, 24);
        var other = Testing.Rand<float>(1, 3, 24, 24);
        var in0Var = MakeVar(input0);
        var in1Var = MakeVar(input1);
        var otherVar = MakeVar(other);
        var add0 = in0Var + otherVar;
        var add1 = in1Var + otherVar;
        var lhs = MakeVar(add0);
        var rhs = MakeVar(add1);
        var mm = IR.F.Math.MatMul(lhs, rhs);
        var f = new BucketFusion("stackvm", mm, new Var[] { lhs, rhs }, Array.Empty<Var>());
        var c = new Call(f, add0, add1);
        var result = TestMatched<MergePrevCallToFusion>(c, new Dictionary<Var, IValue>
        {
            { in0Var, Value.FromTensor(input0) },
            { in1Var, Value.FromTensor(input1) },
            { otherVar, Value.FromTensor(other) },
        });
        var fusion = GetResultFusion(result);
        Assert.Equal(3, fusion.Parameters.Length);
    }

    [Fact]
    public void TestMergeInputWhichHadBeMerged()
    {
        // fusion(add(input, other), other) -> fusion(input, other)
        var input0 = Testing.Rand<float>(1, 3, 24, 24);
        var other = Testing.Rand<float>(1, 3, 24, 24);
        var in0Var = MakeVar(input0);
        var otherVar = MakeVar(other);
        var add0 = in0Var + otherVar;
        var lhs = MakeVar(add0);
        var rhs = MakeVar(otherVar);
        var mm = IR.F.Math.MatMul(lhs, rhs);
        var f = new BucketFusion("stackvm", mm, new Var[] { lhs, rhs }, Array.Empty<Var>());
        var c = new Call(f, add0, otherVar);
        var result = TestMatched<MergePrevCallToFusion>(c, new Dictionary<Var, IValue>
        {
            { in0Var, Value.FromTensor(input0) },
            { otherVar, Value.FromTensor(other) },
        });
        var fusion = GetResultFusion(result);
        Assert.Equal(2, fusion.Parameters.Length);
    }

    // avoid
    // fusion -> nextCall -> nextCallUser -> user1
    //                                    -> user2
    // 这种情况在合并nextCall后，如果nextCallUser被匹配到了，那么此时只会有一个user，因此暂时不rewrite
    [Fact]
    public void TestMergeNextWithUserHasMultiUser()
    {
        var input0 = Testing.Rand<float>(1, 3, 24, 24);
        var in0Var = new Var(new TensorType(input0.ElementType, input0.Shape));
        var a = MakeSingleSimpleFusionCall(Abs, Softmax(in0Var, 0));
        var s = Sqrt(a);
        var sin = Sin(s);
        var e = Exp(sin);
        var f = Floor(sin);
        var body = e + f;
        var newBody = TestMatched<MergeNextCallToFusion>(body, new Dictionary<Var, IValue> { { in0Var, Value.FromTensor(input0) } });
        var c = new FusionCounterVisitor();
        c.Visit(newBody);
        Assert.Equal(1, c.Count);
        TestNotMatch<MergeNextCallToFusion>(newBody);
    }

    [Fact]
    public void TestMergeInputInTupleWhichHadBeMerged()
    {
        var lhs = new Var(new TensorType(DataTypes.Int32, new[] { 1 }));
        var rhs = new Var(new TensorType(DataTypes.Int32, new[] { 2 }));
        var bn = lhs + rhs;
        var f = new BucketFusion("stackvm", bn, new Var[] { lhs, rhs }, Array.Empty<Var>());

        var input = new Var(new TensorType(DataTypes.Int32, Shape.Scalar));
        var r = Reshape(input, new[] { 1 });
        var concat = Concat(new IR.Tuple(r, (Expr)new[] { 1 }), 0);
        var c = new Call(f, r, concat);
        var result = TestMatched<MergePrevCallToFusion>(c, new Dictionary<Var, IValue>
        {
            { input, Value.FromTensor(2) },
        });
        var call = (Call)result;
        var fusion = (BucketFusion)call.Target;
        Assert.Equal(1, fusion.Parameters.Length);
        Assert.Equal(1, call.Arguments.Length);
    }

    [Fact]
    public async Task TestFoldNopTuple()
    {
        var input = (Expr)new[] { 1, 3, 24, 48 };
        var t = new IR.Tuple(new[] { input[0], input[1] });
        var b = t[0] + t[1];
        Dumpper.DumpIR(b, "b");
        var result = await new FoldNopTuple().RunAsync(new Function(b), new());
        Dumpper.DumpIR(result, "result");
    }

    private static BucketFusion GetResultFusion(Expr result)
    {
        var fusion = (BucketFusion)((Call)result).Target;
        return fusion;
    }

    private static Var MakeVar(Tensor<float> mainInput0) => new("input", new TensorType(mainInput0.ElementType, mainInput0.Shape));

    private static Var MakeVar(Expr expr) => new("input", expr.CheckedType);
}
