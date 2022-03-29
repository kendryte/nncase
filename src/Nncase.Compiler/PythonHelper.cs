namespace Nncase.Compiler;

public static class PythonHelper
{
    public static IValue TensorValueFromBytes(DataType type, byte[] span, int[] dimensions)
    {
        return Value.FromTensor(Tensor.FromBytes(type, span, dimensions));
    }

    public static byte[] BytesBufferFromTensor(Tensor value)
    {
        return value.BytesBuffer.ToArray();
    }
}