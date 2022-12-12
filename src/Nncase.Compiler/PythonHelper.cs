using System.Diagnostics;
using System.Runtime.InteropServices;
using NetFabric.Hyperlinq;
using Nncase.Evaluator;
using Nncase.IR;
using Nncase.IR.Math;
using Nncase.Quantization;
using Nncase.Runtime.Interop;
using Nncase.Utilities;

namespace Nncase.Compiler;

public static class PythonHelper
{
    public static void LaunchDebugger()
    {
        Console.WriteLine(System.Environment.Version.ToString());
        Debugger.Launch();
    }

    public static IValue TensorValueFromBytes(DataType type, byte[] span, int[] dimensions)
    {
        return Value.FromTensor(Tensor.FromBytes(type, span, dimensions));
    }

    public static Tensor TensorFromBytes(DataType type, byte[] span, int[] dimensions)
    {
        return Tensor.FromBytes(type, span, dimensions);
    }

    public static byte[] BytesBufferFromTensor(Tensor value)
    {
        return value.BytesBuffer.ToArray();
    }

    public static Memory<byte> ToMemory(byte[] bytes) => new(bytes);

    public static byte[] GetRTTensorBytes(RTTensor tensor)
    {
        var buffer = tensor.Buffer.Buffer.AsHost()!;
        using (var mmOwner = buffer.Map(RTMapAccess.Read))
        {
            return mmOwner.Memory.Span.ToArray();
        }
    }

    public static uint[] GetRTTensorDims(RTTensor tensor)
    {
        return tensor.Dimensions.ToArray();
    }

    public static IValue Evaluate(Expr expr, IReadOnlyDictionary<Var, IValue>? varsValues = null)
    {
        if (CompilerServices.CompileOptions.DumpLevel > 4)
            return DumpManager.RunWithDump("Evaluator", () => CompilerServices.Evaluate(expr, varsValues));
        else
            return CompilerServices.Evaluate(expr, varsValues);
    }

    public static RTTensor[] RunSimulator(RTInterpreter interp, RTValue[] input)
    {
        interp.SetDumpRoot(CompilerServices.CompileOptions.DumpDir);
        var entry = interp.Entry;
        var result = entry.Invoke(input);
        if (result is RTTensor tensor)
        {
            return new[] { tensor };
        }
        else if (result is RTTuple tuple)
        {
            // todo: field maybe a tuple, but not process in this
            return tuple.Fields.Select(x => (RTTensor)x).ToArray();
        }

        throw new NotImplementedException();
    }

    public static bool TargetExist(String target)
    {
        try
        {
            CompilerServices.GetTarget(target);
            return true;
        }
        catch (Exception e)
        {
            Console.WriteLine(e);
            return false;
        }
    }

    // Tensor[sample_count * input_count] dataSet 
    public static PytestCalibrationDatasetProvider MakeDatasetProvider(Tensor[] dataSet, int sampleCount, Var[] fnParams)
    {
        var inputCount = dataSet[0].Length / sampleCount;

        var samples = dataSet.Chunk(inputCount).Select(inputs => inputs.Zip(fnParams).ToDictionary(
            item => item.Item2,
            item => (IValue)Value.FromTensor(item.Item1))).ToArray().ToAsyncEnumerable();
        return new PytestCalibrationDatasetProvider(samples, sampleCount);
    }


    public class PytestCalibrationDatasetProvider : ICalibrationDatasetProvider
    {
        public int? Count => SampleCount;

        public IAsyncEnumerable<IReadOnlyDictionary<Var, IValue>> Samples { get; }

        private int SampleCount = 0;

        public PytestCalibrationDatasetProvider(IAsyncEnumerable<IReadOnlyDictionary<Var, IValue>> samples, int sampleCount)
        {
            Samples = samples;
            SampleCount = sampleCount;
        }
    }
}