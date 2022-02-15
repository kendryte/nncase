using System.Numerics.Tensors;
using System.Runtime.InteropServices;

namespace Nncase.Simulator
{


    public class RuntimeTensor : IDisposable
    {

        public IntPtr Handle { get; private set; }

        public IntPtr memPtr { get; private set; }

        private bool disposedValue;

        [DllImport("libnncase_csharp")]
        static extern unsafe IntPtr RuntimeTensor_from_buffer(
          [In] IntPtr buffer_ptr, PrimTypeCode datatype,
          [In] int* shape_ptr, int shape_size,
          nuint total_items, nuint item_size,
          [In] int* stride_ptr);

        [DllImport("libnncase_csharp")]
        static extern void RuntimeTensor_free(IntPtr rt);

        [DllImport("libnncase_csharp")]
        static extern unsafe void RuntimeTensor_to_buffer(IntPtr rt,
                 [In] byte* buffer_ptr, ref PrimTypeCode datatype);


        [DllImport("libnncase_csharp")]
        static extern unsafe void RuntimeTensor_copy_to([In, Out] IntPtr from, [In, Out] IntPtr dest);


        [DllImport("libnncase_csharp")]
        static extern unsafe int RuntimeTensor_shape(IntPtr rt, int* shape_ptr);

        [DllImport("libnncase_csharp")]
        static extern unsafe int RuntimeTensor_strides(IntPtr rt, int* strides_ptr);

        [DllImport("libnncase_csharp")]
        static extern unsafe PrimTypeCode RuntimeTensor_dtype(IntPtr rt);

        /// <summary>
        /// the default ctor
        /// </summary>
        public RuntimeTensor()
        {
            Handle = IntPtr.Zero;
            memPtr = IntPtr.Zero;
        }

        /// <summary>
        /// get the shape
        /// </summary>
        public unsafe int[] Shape
        {
            get
            {
                if (Handle == IntPtr.Zero)
                {
                    throw new InvalidOperationException("This Tensor Have No Handle");
                }
                var len = RuntimeTensor_shape(Handle, (int*)0);
                var shape = new int[len];
                fixed (int* shape_ptr = shape)
                {
                    RuntimeTensor_shape(Handle, shape_ptr);
                }
                return shape;
            }
        }

        /// <summary>
        /// get strides
        /// </summary>
        public unsafe int[] Strides
        {
            get
            {
                if (Handle == IntPtr.Zero)
                {
                    throw new InvalidOperationException("This Tensor Have No Handle");
                }
                var len = RuntimeTensor_strides(Handle, (int*)0);
                var strides = new int[len];
                fixed (int* strides_ptr = strides)
                {
                    RuntimeTensor_strides(Handle, strides_ptr);
                }
                return strides;
            }
        }

        public DataType DType
        {
            get
            {
                return new PrimType(RuntimeTensor_dtype(Handle));
            }
        }

        int compute_size(ReadOnlySpan<int> shape, ReadOnlySpan<int> strides)
        {
            int max_stride = 0, max_shape = 0;
            for (int i = 0; i < shape.Length; i++)
            {
                if ((shape[i] == 1 ? 0 : strides[i]) > max_stride)
                {
                    max_stride = strides[i];
                    max_shape = shape[i];
                }
            }
            int size = max_stride * max_shape;
            return size != 0 ? size : 1;
        }

        public int Length
        {
            get
            {
                return compute_size(Shape, Strides);
            }
        }

        /// <summary>
        /// create the RuntimeTensor from the DenseTensor
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="tensor"></param>
        /// <returns></returns>
        public static unsafe RuntimeTensor Create<T>(DenseTensor<T> tensor)
          where T : unmanaged, System.IEquatable<T>
        {
            var dtype = DataTypes.FromType<T>();
            var total_bytes = tensor.Length * DataTypes.GetLength(dtype);
            var memPtr = Marshal.AllocHGlobal((int)total_bytes);

            MemoryMarshal.AsBytes(tensor.Buffer.Span).CopyTo(
              new Span<byte>((void*)memPtr, (int)total_bytes));

            fixed (int* shape_ptr = tensor.Dimensions)
            {
                fixed (int* stride_ptr = tensor.Strides)
                {
                    var impl = RuntimeTensor_from_buffer(memPtr, dtype.TypeCode, shape_ptr, tensor.Dimensions.Length, (nuint)tensor.Length, (nuint)DataTypes.GetLength(dtype), stride_ptr);
                    return new RuntimeTensor()
                    {
                        Handle = impl,
                        memPtr = memPtr
                    };
                }
            }
        }

        /// <summary>
        /// convert the runtime tensor to dense tensor.
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <returns></returns>
        /// <exception cref="InvalidCastException"></exception>
        public unsafe DenseTensor<T> ToDense<T>()
          where T : unmanaged
        {
            if (DataTypes.FromType<T>() == DType)
            {
                var tensor = new DenseTensor<T>(Length);
                new Span<T>(memPtr.ToPointer(), Length).CopyTo(tensor.Buffer.Span);
                return tensor;
            }
            throw new InvalidCastException($"The Tensor Type Is {DType}");
        }

        public void CopyTo(RuntimeTensor dest)
        {
            RuntimeTensor_copy_to(Handle, dest.Handle);
        }

        protected virtual void Dispose(bool disposing)
        {
            if (!disposedValue)
            {
                if (disposing)
                {
                    // TODO: dispose managed state (managed objects)
                }

                // TODO: free unmanaged resources (unmanaged objects) and override finalizer
                Marshal.FreeHGlobal(memPtr);
                memPtr = IntPtr.Zero;
                RuntimeTensor_free(Handle);
                Handle = IntPtr.Zero;
                // TODO: set large fields to null
                disposedValue = true;
            }
        }

        // TODO: override finalizer only if 'Dispose(bool disposing)' has code to free unmanaged resources
        ~RuntimeTensor()
        {
            // Do not change this code. Put cleanup code in 'Dispose(bool disposing)' method
            Dispose(disposing: false);
        }

        public void Dispose()
        {
            // Do not change this code. Put cleanup code in 'Dispose(bool disposing)' method
            Dispose(disposing: true);
            GC.SuppressFinalize(this);
        }


        /// <summary>
        /// create the runtime tensor by handle
        /// </summary>
        /// <param name="handle"></param>
        /// <returns></returns>
        public static RuntimeTensor Create(IntPtr handle)
        {
            return new RuntimeTensor()
            {
                Handle = handle,
                memPtr = IntPtr.Zero
            };
        }
    }
}