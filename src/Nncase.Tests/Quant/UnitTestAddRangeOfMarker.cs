// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using NetFabric.Hyperlinq;
using Nncase.IR;
using Nncase.Passes;
using Nncase.Passes.Rules.Neutral;
using Nncase.Quantization;
using Nncase.Tests.TestFixture;
using Xunit;
using static Nncase.IR.F.NN;

namespace Nncase.Tests.QuantTest;

[AutoSetupTestMethod(InitSession = true)]
public class UnitTestAddRangeOfMarker : TestClassBase
{
    [Fact]
    public async Task TestAddRangeOfMarkerToLeaky()
    {
        var input = new Var("input", new TensorType(DataTypes.Float32, new[] { 1, 3, 224, 224 }));
        var leaky = LeakyRelu(input, 0.1);
        var output = leaky;
        var dumpVisitor = await TestAddRangeOfMarkerMainPassesAsync(input, output);

        Assert.Equal(-1.0001221f, ((TensorConst)dumpVisitor.ExprMemo.Keys.ToList()[2]).Value.ToArray<float>()[0]);
        Assert.Equal(1.0001087f, ((TensorConst)dumpVisitor.ExprMemo.Keys.ToList()[2]).Value.ToArray<float>()[1]);
        Assert.Equal(-0.100067139f, ((TensorConst)dumpVisitor.ExprMemo.Keys.ToList()[6]).Value.ToArray<float>()[0]);
        Assert.Equal(1.00005388f, ((TensorConst)dumpVisitor.ExprMemo.Keys.ToList()[6]).Value.ToArray<float>()[1]);
    }

    [Fact]
    public async Task TestAddRangeOfMarkerToRelu6()
    {
        var input = new Var("input", new TensorType(DataTypes.Float32, new[] { 1, 3, 224, 224 }));
        var relu6 = Relu6(input);
        var output = relu6;
        var dumpVisitor = await TestAddRangeOfMarkerMainPassesAsync(input, output);

        Assert.Equal(-1.0001221f, ((TensorConst)dumpVisitor.ExprMemo.Keys.ToList()[2]).Value.ToArray<float>()[0]);
        Assert.Equal(1.0001087f, ((TensorConst)dumpVisitor.ExprMemo.Keys.ToList()[2]).Value.ToArray<float>()[1]);
        Assert.Equal(-6.103435E-05f, ((TensorConst)dumpVisitor.ExprMemo.Keys.ToList()[5]).Value.ToArray<float>()[0]);
        Assert.Equal(1.0000478f, ((TensorConst)dumpVisitor.ExprMemo.Keys.ToList()[5]).Value.ToArray<float>()[1]);
    }

    [Fact]
    public async Task TestAddRangeOfMarkerToBinary()
    {
        var input = new Var("input", new TensorType(DataTypes.Float32, new[] { 1, 2048, 7, 7 }));
        var v0 = IR.F.Tensors.NCHWToNHWC(input);
        var v1 = v0 * Testing.Rand<float>(2048);
        var output = v1;
        _ = await TestAddRangeOfMarkerMainPassesAsync(input, output);
    }

    private async Task<DumpVisitor> TestAddRangeOfMarkerMainPassesAsync(Var input, Expr output)
    {
        CompileOptions.QuantizeOptions.ModelQuantMode = ModelQuantMode.UsePTQ;
        CompileOptions.QuantizeOptions.QuantType = DataTypes.UInt8;
        CompileOptions.QuantizeOptions.WQuantType = DataTypes.UInt8;

        var module = new IRModule(new Function("main", output, new Var[] { input }));

        var pmgr = CompileSession.CreatePassManager("Passes");

        CompileOptions.QuantizeOptions.CalibrationDataset = new SolidCalibrationDatasetProvider(new Var[] { input });
        CompileOptions.QuantizeOptions.CalibrationMethod = CalibMethod.Kld;

        // 0. TargetIndependentPass
        pmgr.AddWithName<DataflowPass>("TargetInDependent").Configure(p =>
        {
            p.Add<AddRangeOfAndMarker>();
        });

        // 1. AssignRanges
        pmgr.AddWithName<EGraphPassWithQuantize>("AssignRanges");

        await pmgr.RunAsync(module);

        var dumpVisitor = new DumpVisitor();
        dumpVisitor.Visit(module.Functions[0]);
        return dumpVisitor;
    }

    public sealed class DumpVisitor : ExprVisitor<int, IRType>
    {
        protected override int DefaultVisitLeaf(Expr expr) => 0;
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
