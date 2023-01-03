// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Text;
using System.Threading.Tasks;
using Nncase;
using Nncase.IR;
using Nncase.Quantization;
using Nncase.TestFixture;
using Nncase.Transform;
using Nncase.Transform.Rules.Neutral;
using Nncase.Utilities;
using Xunit;
using static Nncase.IR.F.NN;
using static Nncase.IR.F.Random;
using static Nncase.IR.F.Tensors;
using static Nncase.PatternMatch.F.Math;
using static Nncase.PatternMatch.Utility;
using static Nncase.Quantization.Utility;
using static Nncase.TestFixture.DataGenerator;
using Random = Nncase.IR.F.Random;

namespace Nncase.Tests.QuantTest;

public class UnitTestKLQuant : TestFixture.UnitTestFixtrue
{
    [Fact]
    public void TestQuantFunction()
    {
        var range = new ValueRange<float>(-1.234f, 2.345f);
        var qp = QuantUtility.GetQuantParam(range, 8, QuantMode.UnsignedMode);
        Assert.Equal(0.014035294f, qp.Scale);
        Assert.Equal(88, qp.ZeroPoint);
    }

    [Fact]
    public void TestKLQuant()
    {
        var caseOptions = GetPassOptions();
        var compileOptions = caseOptions.CompileOptions;
        compileOptions.ModelQuantMode = ModelQuantMode.UsePTQ;
        compileOptions.QuantType = DataTypes.UInt8;
        compileOptions.WQuantType = DataTypes.UInt8;

        Transform.RunPassContext passOptions = new(compileOptions);
        _ = CompilerServices.GetTarget(compileOptions.Target);

        var input = new Var("input", new TensorType(DataTypes.Float32, new[] { 1, 3, 224, 224 }));

        var weightsValue = new List<float>();
        for (int i = 0; i < 16 * 3 * 3 * 3; i++)
        {
            weightsValue.Add(((i * 1.0f / (16 * 3 * 3 * 3)) - 0.5f) * 2);
        }

        var biasValue = new List<float>();
        for (int i = 0; i < 16; i++)
        {
            biasValue.Add(((i * 1.0f / 16) - 0.5f) * 2);
        }

        var weights = Tensor.From<float>(weightsValue.ToArray(), new[] { 16, 3, 3, 3 });
        var bias = Tensor.From<float>(biasValue.ToArray(), new[] { 16 });
        var stride = Tensor.From<int>(new[] { 1, 1 }, new[] { 2 });
        var dilation = Tensor.From<int>(new[] { 1, 1 }, new[] { 2 });
        var padding = new[] { new[] { 0, 1 }, new[] { 0, 0 } };

        var conv = IR.F.NN.Conv2D(input, weights, bias, stride, Pad(padding), dilation, PadMode.Constant, 1);

        var output = conv;
        var module = new IRModule(new Function("main", output, new Var[] { input }));
        _ = (weights.Shape.Size / weights.Shape[0]).FixedValue;

        Transform.PassManager pmgr = new(module, passOptions);

        compileOptions.QuantizeOptions = new()
        {
            CalibrationDataset = new SolidCalibrationDatasetProvider(new Var[] { input }),
            CalibrationMethod = CalibMethod.Kld,
        };

        // 0. TargetIndependentPass
        pmgr.Add(new Transform.DataflowPass("0_TargetInDependent")
        {
            new Transform.Rules.Neutral.AddRangeOfAndMarkerToConv2D(),
        });

        // 1. AssignRanges
        pmgr.Add(new Quantization.EGraphPassWithQuantize("1_AssignRanges", compileOptions.QuantizeOptions!));

        pmgr.RunAsync();

        System.Console.WriteLine(CompilerServices.Print((Function)module.Functions[0]));
        var dumpVisitor = new DumpVisitor();
        dumpVisitor.Visit(module.Functions[0]);

        Assert.Equal(((TensorConst)dumpVisitor.ExpressionMemo.Keys.ToList()[2]).Value.ToArray<float>()[0], -1.0001221f);
        Assert.Equal(1.0001087f, ((TensorConst)dumpVisitor.ExpressionMemo.Keys.ToList()[2]).Value.ToArray<float>()[1]);
        Assert.Equal(((TensorConst)dumpVisitor.ExpressionMemo.Keys.ToList()[5]).Value.ToArray<float>()[0], -1.0001218f);
        Assert.Equal(0.9954922f, ((TensorConst)dumpVisitor.ExpressionMemo.Keys.ToList()[5]).Value.ToArray<float>()[1]);
        Assert.Equal(((TensorConst)dumpVisitor.ExpressionMemo.Keys.ToList()[13]).Value.ToArray<float>()[0], -8.882528f);
        Assert.Equal(9.717726f, ((TensorConst)dumpVisitor.ExpressionMemo.Keys.ToList()[13]).Value.ToArray<float>()[1]);
    }

    private Expr Pad(int[][] p) => Const.FromTensor(Tensor.From<int>(p.SelectMany(i => i).ToArray(), new[] { 2, 2 }));

    public sealed class DumpVisitor : ExprVisitor<int, IRType>
    {
        public override int DefaultVisitLeaf(Expr expr) => 0;

        public override object DefaultVisitLeaf(IVisitable visitable) => 0;

        public int FoundOpCount<T>()
          where T : Op
        {
            return ExpressionMemo.Keys.OfType<T>().Count();
        }
    }

    internal sealed class RandCalibrationDatasetProvider : ICalibrationDatasetProvider
    {
        private const int CountValue = 5;

        public RandCalibrationDatasetProvider(IEnumerable<Var> vars)
        {
            Samples = Enumerable.Range(0, CountValue).Select(i =>
            {
                var values = new Dictionary<Var, IValue>();
                foreach (var var in vars)
                {
                    CompilerServices.InferenceType(var);
                    var shape = var.CheckedShape.Select(d => d.IsUnknown ? 1 : d.FixedValue).ToArray();
                    var value = Value.FromTensor(IR.F.Random.Normal(var.CheckedDataType, 0, 1, 0, shape).Evaluate()
                        .AsTensor());
                    values.Add(var, value);
                }

                return values;
            }).ToAsyncEnumerable();
        }

        public int? Count => CountValue;

        public IAsyncEnumerable<IReadOnlyDictionary<Var, IValue>> Samples { get; }
    }

    internal sealed class SolidCalibrationDatasetProvider : ICalibrationDatasetProvider
    {
        private const int CountValue = 5;

        public SolidCalibrationDatasetProvider(IEnumerable<Var> vars)
        {
            Samples = Enumerable.Range(0, CountValue).Select(i =>
            {
                var values = new Dictionary<Var, IValue>();
                foreach (var var in vars)
                {
                    CompilerServices.InferenceType(var);
                    var shape = var.CheckedShape.Select(d => d.IsUnknown ? 1 : d.FixedValue).ToArray();

                    var shapeSize = 1;
                    for (int j = 0; j < shape.Length; j++)
                    {
                        shapeSize *= shape[j];
                    }

                    var tmpValue = new List<float>();
                    for (int j = 0; j < shapeSize; j++)
                    {
                        tmpValue.Add(((j * 1.0f / shapeSize) - 0.5f) * 2);
                    }

                    var value = Value.FromTensor(Tensor.From<float>(tmpValue.ToArray(), shape));
                    values.Add(var, value);
                }

                return values;
            }).ToAsyncEnumerable();
        }

        public int? Count => CountValue;

        public IAsyncEnumerable<IReadOnlyDictionary<Var, IValue>> Samples { get; }
    }
}
