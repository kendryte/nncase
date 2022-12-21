using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Text;
using System.Threading.Tasks;
using Nncase.Transform;
using Nncase.Transform.Rules.Neutral;
using Xunit;
using Random = Nncase.IR.F.Random;

using Nncase;
using Nncase.IR;
using Nncase.Quantization;
using Nncase.TestFixture;
using Nncase.Utilities;
using System.Collections.Immutable;
using static Nncase.IR.F.NN;
using static Nncase.IR.F.Random;
using static Nncase.IR.F.Tensors;
using static Nncase.PatternMatch.F.Math;
using static Nncase.PatternMatch.Utility;
using static Nncase.Quantization.Utility;
using static Nncase.TestFixture.DataGenerator;

namespace Nncase.Tests.QuantTest;

public class UnitTestKLQuant : TestFixture.UnitTestFixtrue
{
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
        private const int count = 5;
        public int? Count => count;

        public IAsyncEnumerable<IReadOnlyDictionary<Var, IValue>> Samples { get; }

        public RandCalibrationDatasetProvider(IEnumerable<Var> vars)
        {
            Samples = Enumerable.Range(0, count).Select(i =>
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
    }

    [Fact]
    public void TestQuantFunction()
    {
        var range = new ValueRange<float>(-1.234f, 2.345f);
        var qp = QuantUtility.GetQuantParam(range, 8, QuantMode.UnsignedMode);
        System.Diagnostics.Debug.Assert(qp.Scale == 0.014035294f);
        System.Diagnostics.Debug.Assert(qp.ZeroPoint == 88);
    }
    Expr Pad(int[][] p) => Const.FromTensor(Tensor.From<int>(p.SelectMany(i => i).ToArray(), new[] { 2, 2 }));

    //[Fact]
    public async Task TestKLQuant()
    {
        var caseOptions = GetPassOptions();
        var compileOptions = caseOptions.CompileOptions;
        compileOptions.ModelQuantMode = ModelQuantMode.UsePTQ;
        compileOptions.QuantType = DataTypes.UInt8;
        compileOptions.WQuantType = DataTypes.UInt8;

        Transform.RunPassOptions passOptions = new(compileOptions);
        var target = CompilerServices.GetTarget(compileOptions.Target);

        Var input = new Var("input", new TensorType(DataTypes.Float32, new[] { 1, 3, 224, 224 }));

        var weights = IR.F.Random.Normal(DataTypes.Float32, 0, 2, 1, new[] { 16, 3, 3, 3 }).Evaluate().AsTensor();
        var bias = IR.F.Random.Normal(DataTypes.Float32, 0, 2, 1, new[] { 16 }).Evaluate().AsTensor();
        var stride = Tensor.From<int>(new[] { 1, 1 }, new[] { 2 });
        var dilation = Tensor.From<int>(new[] { 1, 1 }, new[] { 2 });
        var padding = new[] { new[] { 0, 1 }, new[] { 0, 0 } };

        var conv = IR.F.NN.Conv2D(input, weights, bias, stride, Pad(padding), dilation, PadMode.Constant, 1);

        var output = conv;
        var module = new IRModule(new Function("main", output, new Var[] { input }));

        var eachChannelSize = (weights.Shape.Size / weights.Shape[0]).FixedValue;

        Transform.PassManager pmgr = new(module, passOptions);

        compileOptions.QuantizeOptions = new()
        {
            CalibrationDataset = new RandCalibrationDatasetProvider(new Var[] { input }),
            CalibrationMethod = CalibMethod.Kld,
        };
        // 0. TargetIndependentPass
        pmgr.Add(new Transform.DataflowPass("0_TargetInDependent")
        {
            new Transform.Rules.Neutral.AddRangeOfAndMarkerToConv2D(),
        });
        // 1. AssignRanges
        pmgr.Add(new Quantization.EGraphPassWithQuantize("1_AssignRanges", compileOptions.QuantizeOptions!));

        await pmgr.RunAsync();

        System.Console.WriteLine(CompilerServices.Print((Function)module.Functions[0]));
        var dumpVisitor = new DumpVisitor();
        dumpVisitor.Visit(module.Functions[0]);

        Assert.Equal(-4.0972834f, ((TensorConst)dumpVisitor.ExpressionMemo.Keys.ToList()[2]).Value.Cast<float>()[0]);
        Assert.Equal(4.206657f, ((TensorConst)dumpVisitor.ExpressionMemo.Keys.ToList()[2]).Value.Cast<float>()[1]);
        Assert.Equal(-6.596074f, ((TensorConst)dumpVisitor.ExpressionMemo.Keys.ToList()[5]).Value.Cast<float>()[0]);
        Assert.Equal(6.5355535f, ((TensorConst)dumpVisitor.ExpressionMemo.Keys.ToList()[5]).Value.Cast<float>()[1]);
        Assert.Equal(-52.367207f, ((TensorConst)dumpVisitor.ExpressionMemo.Keys.ToList()[13]).Value.Cast<float>()[0]);
        Assert.Equal(50.035152f, ((TensorConst)dumpVisitor.ExpressionMemo.Keys.ToList()[13]).Value.Cast<float>()[1]);
    }
}
