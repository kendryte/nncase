using Nncase.Runtime.Interop;

namespace Nncase.Compiler;

public static class PythonHelper
{
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

    public static RTHostBuffer RunSimulator(RTInterpreter interp, RTValue[] input)
    {
        var entry = interp.Entry;
        var result = (RTTensor)entry.Invoke(input);
        var buffer = result.Buffer.Buffer.AsHost()!;
        return buffer;
    }
}