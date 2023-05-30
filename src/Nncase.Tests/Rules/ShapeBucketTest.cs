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
using Nncase.Quantization;
using Nncase.Tests.TestFixture;
using Nncase.Tests.TransformTest;
using Nncase.Utilities;
using Xunit;
using Xunit.Abstractions;
using static Nncase.IR.F.Math;
using static Nncase.IR.F.Tensors;

namespace Nncase.Tests.Rules;

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

    private Var Scalar(string name) => new Var(new TensorType(DataTypes.Int32, Shape.Scalar));
}
