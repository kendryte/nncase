using Nncase.IR;
using System;
using System.Linq;
using System.Collections.Generic;

namespace Nncase.TIR
{

    /// <summary>
    /// The buffer
    /// <example>
    ///     Here's an example of how broadcast buffer can be used to define a symbolic broadcast operation,
    ///   <code>
    ///        m0, m1, m2 = te.var("m0"), te.var("m1"), te.var("m2")
    ///        n0, n1, n2 = te.var("n0"), te.var("n1"), te.var("n2")
    ///        o0, o1, o2 = te.var("o0"), te.var("o1"), te.var("o2")
    ///        A = te.placeholder((m0, m1, m2), name='A')
    ///        B = te.placeholder((n0, n1, n2), name='B')
    ///        C = te.compute((o0, o1, o2), lambda i, j, k: A[i, j, k] + B[i, j, k], name='C')
    ///        Ab = tvm.tir.decl_buffer(A.shape, A.dtype, name="Ab", buffer_type="auto_broadcast")
    ///        Bb = tvm.tir.decl_buffer(B.shape, B.dtype, name="Bb", buffer_type="auto_broadcast")
    ///        s = te.create_schedule(C.op)
    ///        fadd = tvm.build(s, [A, B, C], target='llvm', name='bcast_add', binds={A:Ab, B:Bb})
    ///        dev = tvm.cpu(0)
    ///        a = tvm.nd.array(np.random.uniform(size=(2, 4, 3)).astype(A.dtype), dev)
    ///        b = tvm.nd.array(np.random.uniform(size=(2, 1, 3)).astype(B.dtype), dev)
    ///        c = tvm.nd.array(np.zeros((2, 4, 3), dtype=C.dtype), dev)
    ///        fadd(a, b, c)
    ///        tvm.testing.assert_allclose(c.numpy(), a.numpy() + b.numpy())
    ///   </code>
    /// </example> 
    /// <remarks>
    ///     Buffer data structure reflects the DLTensor structure in dlpack.
    ///     While DLTensor data structure is very general, it is usually helpful
    ///     to create function that only handles specific case of data structure
    ///     and make compiled function benefit from it.
    ///     
    ///     If user pass strides and elem_offset is passed as None
    ///     when constructing the function, then the function will be specialized
    ///     for the DLTensor that is compact and aligned.
    ///     If user pass a fully generic symbolic array to the strides,
    ///     then the resulting function becomes fully generic.
    /// </remarks>
    /// </summary>
    public sealed record Buffer
    {
        /// <summary>
        /// The pointer to the head of the data
        /// <seealso cref="DataAlignment"/>
        /// </summary>
        public Var Handle;

        /// <summary>
        /// data type in the content of the tensor
        /// </summary>
        public DataType Dtype => Handle.CheckedDataType;

        /// <summary>
        /// The shape of the buffer
        /// </summary>
        public IR.Tuple Shape;
        /// <summary>
        /// optional name of the buffer 
        /// </summary>
        public string Name;
        /// <summary>
        /// The strides of each dimension
        ///  This can be an empty array, indicating array is contiguous
        /// </summary>
        public IR.Tuple Strides;
        /// <summary>
        /// The offset in terms of number of dtype elements (including lanes)
        /// </summary>
        public Expr ElemOffset;
        /// <summary>
        /// Alignment requirement of data pointer in bytes.
        /// </summary>
        public int DataAlignment;
        /// <summary>
        /// Factor of elem_offset field,
        ///  elem_offset is guaranteed to be multiple of offset_factor.
        /// </summary>
        public int OffsetFactor;
        /// <summary>
        /// buffer type
        /// </summary>
        public BufferMode BufferMode;

        /// <summary>
        /// Declare a new symbolic buffer.
        /// Normally buffer is created automatically during lower and build.
        /// This is only needed if user want to specify their own buffer layout.
        /// 
        /// See the note below for detailed discussion on usage of buffer.
        ///  <see cref="Buffer"/>
        /// </summary>
        /// <param name="shape">The shape of the buffer.</param>
        /// <param name="dtype">The data type of the buffer.</param>
        /// <param name="name">The name of the buffer.</param>
        /// <param name="data_handle">The data pointer in the buffer.</param>
        /// <param name="strides">The stride of the buffer.</param>
        /// <param name="elem_offset">
        ///   The beginning offset of the array to data.
        ///   In terms of number of elements of dtype.
        /// </param>
        /// <param name="scope">
        ///   The storage scope of the buffer, if not global.
        ///   If scope equals empty string, it means it is global memory.
        /// </param>
        /// <param name="data_alignment">
        ///   The alignment of data pointer in bytes.
        ///   If -1 is passed, the alignment will be set to TVM's internal default.
        /// </param>
        /// <param name="offset_factor">
        ///   The factor of elem_offset field, when set,
        ///   elem_offset is required to be multiple of offset_factor.
        ///   If 0 is pssed, the alignment will be set to 1.
        ///   if non-zero is passed, we will created a Var for elem_offset if elem_offset is not None.
        /// </param>
        /// <param name="buffer_mode">
        ///   auto_broadcast buffer allows one to implement broadcast computation
        ///   without considering whether dimension size equals to one.
        ///   TVM maps buffer[i][j][k] -> buffer[i][0][k] if dimension j's shape equals 1.
        /// </param>
        /// <returns>Buffer</returns>
        public static Buffer Decl(IR.Tuple shape, DataType? dtype = null, string name = "buffer", Var? data_handle = null, IR.Tuple? strides = null, Expr? elem_offset = null, string scope = "", int data_alignment = -1, int offset_factor = 0, BufferMode buffer_mode = BufferMode.Default)
        {
            dtype ??= DataType.Float32;
            strides ??= new();
            if (offset_factor != 0 && elem_offset is null)
            {
                elem_offset = Var.Scalar($"{name}_elem_offset", shape[0].CheckedDataType);
            }
            if (data_handle is null)
            {
                data_handle = Var.Handle(name, dtype, scope);
            }

            elem_offset ??= (Const)0;
            if (data_alignment <= 0)
            {
                data_alignment = 128;
            }
            if (offset_factor == 0)
            {
                offset_factor = 1;
            }

            if (buffer_mode == BufferMode.AutoBroadcast && shape.Count > 0 && strides is null)
            {
                strides = new IR.Tuple(shape.Fields.Select(e => new Var("stride", TensorType.Scalar(e.CheckedDataType))).ToArray());
            }
            return new Buffer(shape, name, data_handle, strides, elem_offset, scope, data_alignment, offset_factor, buffer_mode);
        }

        /// <summary>
        /// <see cref="Decl(IR.Tuple, DataType, string, Var?, IRArray{Expr}?, Expr?, string, int, int, BufferMode)"/>
        /// </summary>
        private Buffer(IR.Tuple shape, string name, Var data, IR.Tuple strides, Expr elem_offset, string scope, int data_alignment, int offset_factor, BufferMode buffer_mode)
        {
            Handle = data;
            Shape = shape;
            Strides = strides;
            Name = name;
            ElemOffset = elem_offset;
            DataAlignment = data_alignment;
            BufferMode = buffer_mode;
            OffsetFactor = offset_factor;
        }

        /// <summary>
        ///  Get an access pointer to the head of buffer.
        ///  This is the recommended method to get buffer data
        ///  ptress when interacting with external functions.
        /// </summary>
        /// <example>
        /// <code>
        ///    // Get access ptr for read
        ///    buffer.access_ptr("r")
        ///    // Get access ptr for read/write with bitmask
        ///    buffer.access_ptr(Buffer.READ | Buffer.WRITE)
        ///    // Get access ptr for read/write with str flag
        ///    buffer.access_ptr("rw")
        ///    // Get access ptr for read with offset
        ///    buffer.access_ptr("r", offset = 100)
        /// </code>
        /// </example>
        /// <param name="access_mode">
        /// The access pattern MASK. Indicate whether the
        /// access will read or write to the data content.
        ///</param>
        /// <param name="content_lanes">
        /// The number of lanes for the data type. This value
        /// is greater than one for vector types.
        /// </param>
        /// <param name="offset">
        /// The offset of pointer. We can use it to offset by
        /// the number of elements from the address of ptr.
        /// </param>
        /// <returns> AccessPtr </returns>
        public Call AccessPtr(AccessMode access_mode, int content_lanes = 1, Expr? offset = null)
        {
            Expr extent;
            offset ??= (Const)0;
            if (Shape.Count == 0)
            {
                extent = (Const)1;
            }
            else if (Strides.Count == Shape.Count)
            {
                extent = Strides[0] * Shape[0] - offset;
            }
            else
            {
                extent = Shape.Aggregate((Expr)1, (a, b) => a * b) - offset;
            }

            Expr elem_offset = ElemOffset + offset;
            var accType = new DataType(Dtype.ElemType, content_lanes);
            if (content_lanes > 1)
            {
                extent = extent / (Const)content_lanes;
                elem_offset = ElemOffset / (Const)content_lanes;
            }

            return new Call(new Builtin.AccessPtr(accType, access_mode), Handle, elem_offset, extent);
        }

        /// <summary>
        /// got the expr of load value from begin index
        /// <remarks>
        /// the only load single value
        /// </remarks>
        /// </summary>
        /// <param name="begin"> the index of data begin</param>
        /// <returns> the corresponding load expr. </returns>
        // public Call VLoad(IR.Tuple begin)
        // {
        //     return F.TOp.Load(Handle, CalcOffset(begin), null);
        // }

        /// <summary>
        /// Generate a Stmt that store value into begin index.
        /// </summary>
        /// <param name="begin">The beginning index in unit of Buffer.dtype</param>
        /// <param name="value">The value to be stored.</param>
        /// <returns>The corresponding store stmt.</returns>
        // public Expr VStore(IR.Tuple begin, Expr value)
        // {
        //     return new Store(Handle, value, CalcOffset(begin), null);
        // }

        public Expr CalcElemOffset(IR.Tuple index)
        {
            var base_offset = ElemOffset;
            if (Strides.Count == 0)
            {
                if (Shape.Count == 0 && index.Count == 1)
                {
                    // if is scalar, only can index 0
                    if (!(index[0] is Const con && con.ToScalar<int>() == 0))
                    {
                        throw new InvalidOperationException($"The Scalar Only Can Index 0, But You Give {index[0]}");
                    }
                }
                else
                {
                    if (Shape.Count != index.Count)
                    {
                        throw new InvalidOperationException($"The Index {index.Count} Not Match Shape {Shape.Count}");
                    }
                    if (index.Count > 0)
                    {
                        var offset = index[0];
                        for (int i = 1; i < index.Count; i++)
                        {
                            offset = (offset * Shape[i]) + index[i];
                        }
                        base_offset = base_offset + offset;
                    }
                }
            }
            else
            {
                if (Strides.Count != index.Count)
                {
                    throw new InvalidOperationException($"The Index {index.Count} Not Match Strides {Strides.Count}");
                }
                if ((Const)0 == base_offset)
                    base_offset = index[0] * Strides[0];
                else
                    base_offset = base_offset + (index[0] * Strides[0]);
                if (index.Count > 0)
                {
                    for (int i = 1; i < index.Count; i++)
                        base_offset = base_offset + (index[i] * Strides[0]);
                }
            }
            return base_offset;
        }

        /// <summary>
        /// compute the load value offset in buffer
        /// </summary>
        /// <param name="index"></param>
        /// <returns></returns>
        public Expr CalcOffset(IR.Tuple index)
        {
            var offset = CalcElemOffset(index);
            return offset;
        }

        /// <summary>
        /// convert the indices to int expr
        /// </summary>
        /// <param name="indices"></param>
        /// <returns> finally index. </returns>
        /// <exception cref="InvalidOperationException"></exception>
        private Expr linearIndices(IEnumerable<Expr> indices)
        {
            if (indices.Count() != Shape.Count)
            {
                throw new InvalidOperationException($"Index Size {indices.Count()} Does Not Match Shape Size {Shape.Count}!");
            }
            Expr dim_size = 1;
            Expr lidx = 0;
            foreach (var (dim, idx) in Enumerable.Zip(Shape.Reverse(), indices.Reverse()))
            {
                lidx = lidx + (idx * dim_size);
                dim_size = dim_size * dim;
            }
            return lidx;
        }
        /// <summary>
        /// Load the Value
        /// </summary>
        /// <param name="indices"> index </param>
        /// <returns> the load Expression. </returns>
        /// <exception cref="InvalidOperationException"></exception>
        public Expr this[params Expr[] indices]
        {
            get
            {
                var index = linearIndices(indices);
                if (Dtype.Lanes > 1)
                {
                    index = F.TOp.Ramp(index * Dtype.Lanes, 1, Dtype.Lanes);
                }
                return F.TOp.Load(Handle, index);
            }
            set
            {
                throw new InvalidOperationException("Your Should Use The Buf.Store([i,j,k],value) For Index Store!");
            }
        }

        /// <summary>
        /// Elem Set, Because we can't return the store expression in the property setter!
        /// </summary>
        /// <param name="indices_with_value"> the last element is value, others are index</param>
        /// <returns> Store expression </returns>
        public Expr Store(params Expr[] indices_with_value)
        {
            if (indices_with_value.Length <= 1)
            {
                throw new InvalidOperationException("The Buffer Store Must Have Index and Value !");
            }
            var index = linearIndices(indices_with_value.SkipLast(1));
            if (Dtype.Lanes > 1)
            {
                index = F.TOp.Ramp(index * Dtype.Lanes, 1, Dtype.Lanes);
            }
            return F.TOp.Store(Handle, indices_with_value.Last(), index);
        }
    }

    public interface DataProducer
    {
        /// <summary>
        /// get shape 
        /// </summary>
        /// <returns> shapes </returns>
        public IRArray<Expr> GetShape();

        /// <summary>
        /// Get the data type of the result.
        /// </summary>
        /// <returns>The data type.</returns>
        public DataType GetDataType();

        /// <summary>
        /// Get the name hint of the data producer.
        /// </summary>
        /// <returns> name string </returns>
        public string GetNameHint();
    }

}