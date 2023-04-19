// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Reflection;
using System.Threading.Tasks;
using NetFabric.Hyperlinq;
using Nncase.IR;
using Nncase.Passes;
using Nncase.Passes.Rules.Neutral;
using Nncase.Quantization;
using Nncase.Tests.TestFixture;
using Xunit;
using static Nncase.IR.F.NN;
using static Nncase.IR.F.Random;

namespace Nncase.Tests.QuantTest;

[AutoSetupTestMethod(InitSession = true)]
public class UnitTestExportQuantScheme : TestClassBase
{
    [Fact]
    public async Task TestExportQuantSchemeForWeightsByTensorConv2D()
    {
        var input = new Var("input", new TensorType(DataTypes.Float32, new[] { 1, 3, 224, 224 }));

        var weightsValue = new List<float>();
        for (int i = 0; i < 32 * 3 * 3 * 3; i++)
        {
            weightsValue.Add(i * 1.0f / (32 * 3 * 3 * 3));
        }

        Expr weights = Tensor.From<float>(weightsValue.ToArray(), new[] { 32, 3, 3, 3 });
        weights.Metadata.OutputNames = new string[] { "weight" };

        var bias = Normal(DataTypes.Float32, new[] { 32 }).Evaluate().AsTensor();
        var stride = Tensor.From(new[] { 1, 1 }, new[] { 2 });
        var dilation = Tensor.From(new[] { 1, 1 }, new[] { 2 });
        var padding = new[,] { { 0, 0 }, { 0, 0 } };

        var conv = Conv2D(input, weights, bias, stride, padding, dilation, PadMode.Constant, 1);

        var output = conv;

        var dumpVisitor = await TestExportQuantSchemeMainPassesAsync(input, output, false);

        var expectedQuantScheme = "{\"Version\":\"1.0\",\"Model\":null,\"Outputs\":[{\"Name\":\"weight\",\"DataType\":\"u8\",\"DataRange\":[{\"Min\":0.0,\"Max\":0.9988426,\"IsFull\":false}],\"DataRangeMode\":\"by_tensor\"}]}";
        Assert.Equal(expectedQuantScheme, CompileOptions.QuantizeOptions.QuantScheme);
    }

    [Fact]
    public async Task TestExportQuantSchemeForWeightsByChannelConv2D()
    {
        var input = new Var("input", new TensorType(DataTypes.Float32, new[] { 1, 3, 224, 224 }));

        var weightsValue = new List<float>();
        for (int i = 0; i < 3 * 3 * 3 * 3; i++)
        {
            weightsValue.Add(i * 1.0f / (3 * 3 * 3 * 3));
        }

        Expr weights = Tensor.From<float>(weightsValue.ToArray(), new[] { 3, 3, 3, 3 });
        weights.Metadata.OutputNames = new string[] { "weight" };

        var bias = Normal(DataTypes.Float32, new[] { 3 }).Evaluate().AsTensor();
        var stride = Tensor.From(new[] { 1, 1 }, new[] { 2 });
        var dilation = Tensor.From(new[] { 1, 1 }, new[] { 2 });
        var padding = new[,] { { 0, 0 }, { 0, 0 } };

        var conv = Conv2D(input, weights, bias, stride, padding, dilation, PadMode.Constant, 1);

        var output = conv;

        var dumpVisitor = await TestExportQuantSchemeMainPassesAsync(input, output, true);

        var expectedQuantScheme = "{\"Version\":\"1.0\",\"Model\":null,\"Outputs\":[{\"Name\":\"weight\",\"DataType\":\"u8\",\"DataRange\":[{\"Min\":0.0,\"Max\":0.32098764,\"IsFull\":false},{\"Min\":0.33333334,\"Max\":0.654321,\"IsFull\":false},{\"Min\":0.6666667,\"Max\":0.9876543,\"IsFull\":false}],\"DataRangeMode\":\"by_channel\"}]}";
        Assert.Equal(expectedQuantScheme, CompileOptions.QuantizeOptions.QuantScheme);
    }

    private async Task<DumpVisitor> TestExportQuantSchemeMainPassesAsync(Var input, Expr output, bool exportWeightRangeByChannel)
    {
        CompileOptions.QuantizeOptions.ModelQuantMode = ModelQuantMode.UsePTQ;
        CompileOptions.QuantizeOptions.QuantType = DataTypes.UInt8;
        CompileOptions.QuantizeOptions.WQuantType = DataTypes.UInt8;

        var module = new IRModule(new Function("main", output, new Var[] { input }));

        var pmgr = CompileSession.CreatePassManager("Passes");

        CompileOptions.QuantizeOptions.CalibrationDataset = new SolidCalibrationDatasetProvider(new Var[] { input });
        CompileOptions.QuantizeOptions.CalibrationMethod = CalibMethod.Kld;
        CompileOptions.QuantizeOptions.ExportQuantScheme = true;
        CompileOptions.QuantizeOptions.ExportWeightRangeByChannel = exportWeightRangeByChannel;

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
