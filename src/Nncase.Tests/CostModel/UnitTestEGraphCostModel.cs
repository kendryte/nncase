// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using Nncase;
using Nncase.CostModel;
using Nncase.IR;
using Nncase.IR.Math;
using Nncase.IR.NN;
using Nncase.Tests.TestFixture;
using Nncase.Transform;
using Xunit;

namespace Nncase.Tests.CostModelTest;

/// <summary>
/// test egraph costs.
/// </summary>
public sealed class UnitTestEGraphCostModel
{
    /// <summary>
    /// root = x + conv2d(x)
    /// root cost = conv2d(x) cost + binary cost.
    /// </summary>
    [Fact]
    public void TestBinaryShortCutCost()
    {
        Tensor GetD<T>(System.IO.BinaryReader reader, long start, int size, params int[] shape)
        where T : unmanaged, IEquatable<T>
        {
            // __reader.BaseStream.Seek(__start, System.IO.SeekOrigin.Begin);
            var bytes = new byte[size];
            Testing.RandGenerator.NextBytes(bytes);
            return Tensor.FromBytes<T>(bytes, shape);
        }

        using var vD = new System.IO.BinaryReader(new System.IO.MemoryStream());
        Function v0; // (f32[1,224,224,3]) -> (f32[1,7,7,2048])
        var v30 = new Var("serving_default_input_1:0", new TensorType(DataTypes.Float32, new[] { 1, 256, 56, 56 }));
        var v30_1 = new Marker("RangeOf", GetD<float>(vD, 269312, 65536, 256, 256, 1, 1), new float[] { -0.3903954f, 0.46443018f }); // f32[256,64,1,1]
        var v30_2 = new Call(new Conv2D(PadMode.Constant), new Expr[] { v30, v30_1, GetD<float>(vD, 334848, 1024, 256), new int[] { 1, 1 }, new int[,] { { 0, 0 }, { 0, 0 } }, new int[] { 1, 1 }, 1, new float[] { -float.PositiveInfinity, float.PositiveInfinity } }); // f32[1,256,56,56]
        var v30_3 = new Marker("RangeOf", v30_2, new float[] { -11.444157f, 10.850164f }); // f32[1,256,56,56]
        var v31 = new Marker("RangeOf", v30_3, new float[] { -35.968758f, 32.784153f }); // f32[1,256,56,56]
        var v46 = new Marker("RangeOf", GetD<float>(vD, 551424, 65536, 256, 256, 1, 1), new float[] { -0.4176304f, 0.74278486f }); // f32[256,64,1,1]
        var v47 = new Call(new Conv2D(PadMode.Constant), new Expr[] { v31, v46, GetD<float>(vD, 616960, 1024, 256), new int[] { 1, 1 }, new int[,] { { 0, 0 }, { 0, 0 } }, new int[] { 1, 1 }, 1, new float[] { -float.PositiveInfinity, float.PositiveInfinity } }); // f32[1,256,56,56]
        var v48 = new Marker("RangeOf", v47, new float[] { -8.493573f, 14.834768f }); // f32[1,256,56,56]
        var v49 = new Call(new Binary(BinaryOp.Add), new Expr[] { v31, v48 }); // f32[1,256,56,56]
        var v50 = new Marker("RangeOf", v49, new float[] { -35.823486f, 31.693804f }); // f32[1,256,56,56]
        v0 = new Function("main", v50, new Var[] { v30 });
        var binaryCost = CompilerServices.EvaluateCost(v49);
        var binaryLhsCost = CompilerServices.EvaluateCost(v31);
        var binaryRhsCost = CompilerServices.EvaluateCost(v48);

        var eGraph = new EGraph();
        var lhsClass = eGraph.Add(v31);
        var rhsClass = eGraph.Add(v48);
        var root = eGraph.Add(v50);

        var costModel = new EGraphCostEvaluator(root, null).Evaluate();
        var rootCost = costModel[root.Nodes[0]];
        var binrayOpCost = CompilerServices.EvaluateOpCost((Op)v49.Target, new EvaluatorContext(v49));
        Assert.Equal(binrayOpCost! + binaryRhsCost!, rootCost);
    }

    /// <summary>
    /// root = concat(tuple(a,a,b),1)
    /// root cost = concat op cost + a cost + b cost.
    /// </summary>
    [Fact]
    public void TestTupleCost()
    {
        var v1 = new Var("serving_default_input_1:0", new TensorType(DataTypes.Float32, new[] { 1, 256, 56, 56 }));
        var a = v1 + Testing.Rand<float>(1, 256, 1, 1);
        var b = v1 * Testing.Rand<float>(1, 256, 56, 56);
        var tuple = new IR.Tuple(a, a, b);
        var root = IR.F.Tensors.Concat(tuple, 1);
        root.InferenceType();
        var aCost = CompilerServices.EvaluateCost(a); // 2512,1476,369
        var bCost = CompilerServices.EvaluateCost(b); // 2952,1476,738
        var concatOpCost = CompilerServices.EvaluateOpCost((Op)root.Target, new EvaluatorContext(root));

        var eGraph = new EGraph();
        _ = eGraph.Add(a);
        _ = eGraph.Add(b);
        var rootEclass = eGraph.Add(root);

        var costModel = new EGraphCostEvaluator(rootEclass, null).Evaluate();
        var rootCost = costModel[rootEclass.Nodes[0]];
        Assert.Equal(aCost! + bCost! + concatOpCost!, rootCost);
    }

    /// <summary>
    ///    input
    ///      |
    ///   conv2d
    ///   /    \
    ///   |   relu     =>  can't duplicate conv2d
    ///   |     |
    ///   |   conv2d
    ///    \  /
    ///    add.
    /// </summary>
    [Fact]
    public void TestDuplicteConv2D()
    {
        var input = new Var("serving_default_input_1:0", new TensorType(DataTypes.Float32, new[] { 1, 256, 56, 56 }));

        var v0W = Testing.Rand<float>(64, 256, 1, 1);
        var v0B = Testing.Rand<float>(64);
        var v0 = new Call(new Conv2D(PadMode.Constant), new Expr[] { input, v0W, v0B, new int[] { 1, 1 }, new int[,] { { 0, 0 }, { 0, 0 } }, new int[] { 1, 1 }, 1, new float[] { -float.PositiveInfinity, float.PositiveInfinity } }); // f32[1,64,56,56]
        var v1 = IR.F.NN.Relu6(v0);
        var v2 = new Call(new Conv2D(PadMode.Constant), new Expr[] { v1, Testing.Rand<float>(64, 64, 1, 1), Testing.Rand<float>(64), new int[] { 1, 1 }, new int[,] { { 0, 0 }, { 0, 0 } }, new int[] { 1, 1 }, 1, new float[] { -float.PositiveInfinity, float.PositiveInfinity } }); // f32[1,256,56,56]
        var v3 = v0 + v2;
        var v1Fused = new Call(new Conv2D(PadMode.Constant), new Expr[] { input, v0W, v0B, new int[] { 1, 1 }, new int[,] { { 0, 0 }, { 0, 0 } }, new int[] { 1, 1 }, 1, new float[] { 0, 6 } });
        CompilerServices.InferenceType(v1Fused);
        CompilerServices.InferenceType(v3);

        var eGraph = new EGraph();
        var v1Class = eGraph.Add(v1);
        var v1FuseClass = eGraph.Add(v1Fused);
        var root = eGraph.Add(v3);
        eGraph.Union(v1Class, v1FuseClass);

        var post = eGraph.Extract(root, null);

        var visitor = new TestVisitor();
        visitor.Visit(post);

        Assert.Equal(2, visitor.CountCallOp<Conv2D>());
    }

    internal sealed class EvaluatorContext : Evaluator.ICostEvaluateContext
    {
        private readonly Call _currentCall;

        public EvaluatorContext(Call call)
        {
            _currentCall = call;
        }

        public T GetArgumentType<T>(Op op, ParameterInfo parameter)
            where T : IRType
            => (T)_currentCall[parameter].CheckedType!;

        public T GetReturnType<T>()
            where T : IRType
            => (T)_currentCall.CheckedType!;
    }
}
