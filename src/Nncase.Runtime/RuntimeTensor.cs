using System.Numerics.Tensors;
using System.Runtime.InteropServices;
namespace Nncase.Runtime
{


    [StructLayout(LayoutKind.Sequential)]
    public struct cRuntimeTensor
    {
        public IntPtr impl;
    }

    public class RuntimeTensor
    {

        cRuntimeTensor inner;

        [DllImport("libnncaseruntime_csharp")]
        static extern cRuntimeTensor RuntimeTensor_from_buffer(in byte buffer_ptr, byte datatype,
                                                  in int shape_ptr, int shape_size,
                                                  ulong total_items, ulong item_size,
                                                  in int stride_ptr);

        private RuntimeTensor() { }

        /// <summary>
        /// create the RuntimeTensor from the DenseTensor
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="tensor"></param>
        /// <returns></returns>
        public static RuntimeTensor FromDense<T>(DenseTensor<T> tensor)
          where T : unmanaged
        {
            var dtype = DataTypes.FromType<T>();
            var inner = RuntimeTensor_from_buffer(MemoryMarshal.GetReference(MemoryMarshal.AsBytes(tensor.Buffer.Span)), (byte)dtype.ElemType,
                            MemoryMarshal.GetReference(tensor.Dimensions), tensor.Dimensions.Length, (ulong)tensor.Length,
                            (ulong)DataTypes.GetLength(dtype), MemoryMarshal.GetReference(tensor.Strides));

            return new RuntimeTensor() { inner = inner };
        }
    }
}