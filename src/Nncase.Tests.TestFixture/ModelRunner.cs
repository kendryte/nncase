// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Nncase.Diagnostics;
using Nncase.IR;
using Nncase.Quantization;
using Nncase.Utilities;

namespace Nncase.Tests
{
    public sealed class SelfInputCalibrationDatasetProvider : ICalibrationDatasetProvider
    {
        private readonly int _count = 1;

        private readonly IAsyncEnumerable<IReadOnlyDictionary<Var, IValue>> _samples;

        public SelfInputCalibrationDatasetProvider(IReadOnlyDictionary<Var, IValue> sample)
        {
            _samples = new[] { sample }.ToAsyncEnumerable();
        }

        public int? Count => _count;

        public IAsyncEnumerable<IReadOnlyDictionary<Var, IValue>> Samples => _samples;
    }

    /// <summary>
    /// Used for easy to run a model and generate data about kmodel.
    /// What you should do:
    /// 1. override ModelPath and MakeInputs.
    /// 2. make test entry and call TestRunner.
    ///    [Fact]
    ///    public async Task run() { await TestRunner(); }.
    /// </summary>
    public abstract class ModelRunner : TestClassBase
    {
        public async Task TestRunner()
        {
            CompileOptions.QuantizeOptions.ModelQuantMode = ModelQuantMode.UsePTQ;
            CompileOptions.QuantizeOptions.CalibrationMethod = CalibMethod.NoClip;
            CompileOptions.QuantizeOptions.BindQuantMethod = false;
            CompileOptions.QuantizeOptions.UseSquant = false;
            CompileOptions.DumpFlags = DumpFlags.Rewrite;

            var modelPath = ModelPath();
            CompileOptions.InputFormat = Path.GetExtension(modelPath).Trim('.');
            var m = await CompileSession.Compiler.ImportModuleAsync(CompileOptions.InputFormat, modelPath);
            var types = m.Entry!.ParameterTypes.Select(type => (TensorType)type!).ToArray();
            var shapes = types.Select(x => x.Shape).ToArray();
            var isDynamic = shapes.Any(shape => !shape.IsFixed);
            var inputs = MakeInputs(types);
            CompileSession.Compiler.ImportIRModule(m);

            var f = (Function)m.Entry!;

            var samples = f.Parameters.ToArray().Zip(inputs)
                .ToDictionary(x => x.First, x => (IValue)Value.FromTensor(x.Second));
            CompileOptions.QuantizeOptions.CalibrationDataset = new SelfInputCalibrationDatasetProvider(samples);
            await CompileSession.Compiler.CompileAsync();
            var (kmodelPath, expectKmodel) = Testing.BuildKModel("test.kmodel", m, CompileSession);
            Testing.DumpInterpModel(kmodelPath, inputs, Path.Join(Dumpper.Directory, "interp"));
            var outputs = Testing.RunKModel(expectKmodel, Dumpper.Directory, inputs).AsTensors();
            DumpUtility.WriteKmodelData(inputs, outputs, kmodelPath, Path.Join(Dumpper.Directory, "kmodel_data"), isDynamic);
        }

        /// <summary>
        /// Default generate random data.
        /// you can read input from bin file by BinFileUtil.ReadBinFile().
        /// </summary>
        /// <returns>Inputs.</returns>
        public virtual Tensor[] MakeInputs(TensorType[] types)
        {
            return types.Select(type => Testing.Rand(type.DType, type.Shape.ToValueArray())).ToArray();
        }

        public abstract string ModelPath();
    }
}
