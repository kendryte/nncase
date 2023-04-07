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
public class UnitTestAssignInfoFromConfig : TestClassBase
{
    [Fact]
    public async Task TestAssignInfoToLeaky()
    {
        var input = new Var("input", new TensorType(DataTypes.Float32, new[] { 1, 3, 224, 224 }));
        var leaky = LeakyRelu(input, 0.1);
        var output = leaky;
        output.Metadata.OutputNames = new string[] { "leaky" };
        var dumpVisitor = await TestAssignInfoMainPassesAsync(input, output);
        
        Assert.Equal(0.0f, ((TensorConst)dumpVisitor.ExprMemo.Keys.ToList()[7]).Value.ToArray<float>()[0]);
        Assert.Equal(1.0f, ((TensorConst)dumpVisitor.ExprMemo.Keys.ToList()[7]).Value.ToArray<float>()[1]);
    }

    private async Task<DumpVisitor> TestAssignInfoMainPassesAsync(Var input, Expr output)
    {
        CompileOptions.QuantizeOptions.ModelQuantMode = ModelQuantMode.UsePTQ;
        CompileOptions.QuantizeOptions.QuantType = DataTypes.UInt8;
        CompileOptions.QuantizeOptions.WQuantType = DataTypes.UInt8;

        var module = new IRModule(new Function("main", output, new Var[] { input }));

        var pmgr = CompileSession.CreatePassManager("Passes");

        CompileOptions.QuantizeOptions.CalibrationDataset = new SolidCalibrationDatasetProvider(new Var[] { input });
        CompileOptions.QuantizeOptions.CalibrationMethod = CalibMethod.Kld;
        CompileOptions.QuantizeOptions.ImportConfigFile = System.IO.Directory.GetCurrentDirectory() + "/../../../Quant/config_leaky.json";

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
