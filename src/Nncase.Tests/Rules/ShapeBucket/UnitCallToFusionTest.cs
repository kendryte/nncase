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

namespace Nncase.Tests.Rules.ShapeBucket;

[AutoSetupTestMethod(InitSession = true)]
public class UnitCallToFusionTest : TransformTestBase
{
    [Fact]
    public void TestMarkerCallToFusion()
    {
        var input = Testing.Rand<float>(1, 3, 24, 24);
        var inputVar = new Var(new TensorType(input.ElementType, input.Shape));
        var m1 = new Marker("RangeOf", inputVar, new[] { 0.1f, 0.2f });
        var abs = Abs(m1);
        var m2 = new Marker("RangeOf", abs, new[] { -0.1f, 0.2f });
        var post = TestMatched<UnaryToFusion>(
            m2,
            new Dictionary<Var, IValue> { { inputVar, Value.FromTensor(input) } });
        Assert.True(post is Marker);
        var postCall = (Call)((Marker)post).Target;
        var fusion = (BucketFusion)postCall.Target;
        Assert.True(postCall.Arguments[0] is Marker);
        Assert.True(fusion.Body is Marker);
    }

    [Fact]
    public void TestMutliUserCallToFusion()
    {
        var input = Testing.Rand<float>(1, 3, 24, 24);
        var inputVar = new Var(new TensorType(input.ElementType, input.Shape));
        var abs = Abs(Softmax(inputVar, 0));
        var fusionVar1 = new Var(new TensorType(input.ElementType, input.Shape));
        var c1 = new Call(new BucketFusion("stackvm", fusionVar1 + 1f, new[] { fusionVar1 }, Array.Empty<Var>()), abs);
        var fusionVar2 = new Var(new TensorType(input.ElementType, input.Shape));
        var c2 = new Call(new BucketFusion("stackvm", fusionVar2 - 1f, new[] { fusionVar2 }, Array.Empty<Var>()), abs);
        var body = new IR.Tuple(c1, c2);
        Dumpper.DumpIR(body, "Body");
        TestMatched<MultiUserCallToFusion>(body, new Dictionary<Var, IValue> { { inputVar, Value.FromTensor(input) } });
    }

    [Fact]
    public void TestConcatToFusion()
    {
        var input0 = Testing.Rand<float>(1, 3, 24, 24);
        var inputVar0 = new Var(new TensorType(input0.ElementType, input0.Shape));
        var input1 = Testing.Rand<float>(1, 3, 24, 24);
        var inputVar1 = new Var(new TensorType(input1.ElementType, input1.Shape));
        var input2 = Testing.Rand<float>(1, 3, 24, 24);
        var inputVar2 = new Var(new TensorType(input2.ElementType, input2.Shape));
        var inputs = new[] { inputVar0, inputVar1, inputVar2 }.Select(x => Softmax(x, 0)).ToArray();
        var cat = Concat(new IR.Tuple(inputs), 0);
        TestMatched<MultiUserCallToFusion>(
            cat,
            new Dictionary<Var, IValue>
            {
                { inputVar0, Value.FromTensor(input0) },
                { inputVar1, Value.FromTensor(input1) },
                { inputVar2, Value.FromTensor(input2) },
            });
    }

    [Fact]
    public void TestConcatToFusionWithConst()
    {
        var input0 = Testing.Rand<float>(1, 3, 24, 24);
        var inputVar0 = new Var(new TensorType(input0.ElementType, input0.Shape));
        var input1 = Testing.Rand<float>(1, 3, 24, 24);
        var inputVar1 = new Var(new TensorType(input1.ElementType, input1.Shape));
        var input2 = Testing.Rand<float>(1, 3, 24, 24);
        var inputs = new Expr[] { inputVar0, inputVar1 }.Select(x => (Expr)Softmax(x, 0)).Append(input2).ToArray();
        var cat = Concat(new IR.Tuple(inputs), 0);
        TestMatched<MultiUserCallToFusion>(
            cat,
            new Dictionary<Var, IValue>
            {
                { inputVar0, Value.FromTensor(input0) },
                { inputVar1, Value.FromTensor(input1) },
            });
    }

    [Fact]
    public void TestConcatSingleInputToFusion()
    {
        var input0 = Testing.Rand<float>(1, 3, 24, 24);
        var inputVar0 = new Var(new TensorType(input0.ElementType, input0.Shape));
        var inputs = new[] { inputVar0 }.Select(x => Softmax(x, 0)).ToArray();
        var cat = Concat(new IR.Tuple(inputs), 0);
        TestMatched<MultiUserCallToFusion>(
            cat,
            new Dictionary<Var, IValue>
            {
                { inputVar0, Value.FromTensor(input0) },
            });
    }

    [Fact]
    public void TestReshapeToFusion()
    {
        var input0 = Testing.Rand<float>(1, 3, 24, 24);
        var inputVar0 = new Var(new TensorType(input0.ElementType, input0.Shape));
        var s = Softmax(inputVar0, 0);
        var r = Reshape(s, Require(true, ShapeOf(s)));
        TestMatched<MultiUserCallToFusion>(r, new Dictionary<Var, IValue> { { inputVar0, Value.FromTensor(input0) } });
    }

    [Fact]
    public void TestComplexReshapeToFusion()
    {
        var input0 = Testing.Rand<float>(1, 3, 24, 24);
        var inputVar0 = new Var(new TensorType(input0.ElementType, input0.Shape));
        var s = Softmax(inputVar0, 0);
        var r = Reshape(s, Require(true, ShapeOf(s)));
        TestMatched<MultiUserCallToFusion>(r, new Dictionary<Var, IValue> { { inputVar0, Value.FromTensor(input0) } });
    }

    [Fact]
    public void TestNoNest()
    {
        var input0 = Testing.Rand<float>(1, 3, 24, 24);
        var inputVar0 = new Var(new TensorType(input0.ElementType, input0.Shape));
        var after = TestMatched<MultiUserCallToFusion>(
            Sqrt(Abs(Softmax(inputVar0, 0))),
            new Dictionary<Var, IValue> { { inputVar0, Value.FromTensor(input0) } });

        var v = new FusionCounterVisitor();
        v.Visit(after);
        Assert.Equal(2, v.Count);
        if (after is Call { Target: BucketFusion fusion } c)
        {
            Assert.Equal(1, c.Arguments.Length);
            var n = new FusionCounterVisitor();
            n.Visit(fusion.Body);
            Assert.Equal(0, n.Count);
        }
    }
}
