// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using Nncase.IR;

namespace Nncase.TIR
{
    /// <summary>
    /// The low level memory buffer.
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
        /// The pointer to the head of the data.
        /// <seealso cref="DataAlignment"/>
        /// </summary>
        public Var Handle;

        /// <summary>
        /// data type in the content of the tensor.
        /// </summary>
        public PrimType Dtype => (PrimType)((PointerType)(((TensorType)Handle.TypeAnnotation).DType)).ElemType;

        /// <summary>
        /// The shape of the buffer.
        /// </summary>
        public IR.Tuple Shape;

        /// <summary>
        /// optional name of the buffer.
        /// </summary>
        public string Name;

        /// <summary>
        /// The strides of each dimension (it's elemwise, not bytes.)
        /// This can be an empty array, indicating array is contiguous
        /// </summary>
        public IR.Tuple Strides;

        /// <summary>
        /// The offset in terms of number of dtype elements (including lanes).
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
        /// buffer type.
        /// </summary>
        public BufferMode BufferMode;

        /// <summary>
        /// <see cref="T.DeclBuffer(IR.Tuple, DataType?, string, Var?, IR.Tuple?, Expr?, string, int, int, BufferMode)"/>.
        /// </summary>
        public Buffer(IR.Tuple shape, string name, Var data, IR.Tuple strides, Expr elem_offset, string scope, int data_alignment, int offset_factor, BufferMode buffer_mode)
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
        /// </param>
        /// <param name="content_lanes">
        /// The number of lanes for the data type. This value
        /// is greater than one for vector types.
        /// </param>
        /// <param name="offset">
        /// The offset of pointer. We can use it to offset by
        /// the number of elements from the address of ptr.
        /// </param>
        /// <returns> AccessPtr. </returns>
        public Call AccessPtr(AccessMode access_mode, int content_lanes = 1, Expr? offset = null)
        {
            if (content_lanes != 1)
            {
                throw new NotImplementedException();
            }

            Expr extent;
            offset ??= (Const)0;
            if (Shape.Count == 0)
            {
                extent = (Const)1;
            }
            else if (Strides.Count == Shape.Count)
            {
                extent = (Strides[0] * Shape[0]) - offset;
            }
            else
            {
                extent = Shape.Aggregate((Expr)1, (a, b) => a * b) - offset;
            }

            Expr elem_offset = ElemOffset + offset;
            var accType = Dtype;
            if (content_lanes > 1)
            {
                extent = extent / (Const)content_lanes;
                elem_offset = ElemOffset / (Const)content_lanes;
            }

            return new Call(new Builtin.AccessPtr(accType, access_mode), Handle, elem_offset, extent);
        }

        /// <summary>
        /// Iter Var subscript create BufferLoad.
        /// </summary>
        /// <param name="indices"> index. </param>
        /// <returns> the Bufferload Expression. </returns>
        /// <exception cref="InvalidOperationException"></exception>
        public Expr this[params IterVar[] indices]
        {
            get
            {
                return new BufferLoad(this, indices);
            }
            set
            {
                throw new InvalidOperationException("Your Should Use Like T.Store(Buf[i,j,k],value) !");
            }
        }

        /// <summary>
        /// if use expr index, will create Load.
        /// </summary>
        /// <param name="index"></param>
        /// <returns></returns>
        /// <exception cref="InvalidOperationException"></exception>
        public Expr this[Expr index]
        {
            get
            {
                return T.Load(Handle, index);
            }
            set
            {
                throw new InvalidOperationException("Your Should Use Like T.Store(Buf[i,j,k],value) !");
            }
        }

        /// <summary>
        /// value load.
        /// </summary>
        /// <param name="indices"></param>
        /// <returns></returns>
        public Expr VLoad(IRArray<Expr> indices)
        {
            return T.Load(Handle, LoadOffset(indices));
        }

        /// <summary>
        /// value store.
        /// </summary>
        /// <param name="indices"></param>
        /// <param name="value"></param>
        /// <returns></returns>
        public Expr VStore(IRArray<Expr> indices, Expr value)
        {
            return T.Store(Handle, LoadOffset(indices), value);
        }

        /// <summary>
        /// IndicesOffset only calc the element based offset.
        /// NOTE it's ignore the data's lanes.
        /// </summary>
        /// <param name="indices"> expr indices. </param>
        /// <returns></returns>
        public Expr IndicesOffset(IRArray<Expr> indices)
        {
            var offset = ElemOffset;
            if (indices.Count != Strides.Count || indices.Count != Shape.Count)
            {
                throw new InvalidOperationException("The indices Length Not Equal Stride Or Shape!");
            }

            for (int i = 0; i < indices.Count; i++)
            {
                offset = offset + (indices[i] * Strides[i]);
            }

            return offset;
        }

        /// <summary>
        /// get the load value index
        /// NOTE we will consider about the lanes.
        /// </summary>
        /// <param name="indices"> the indices expr.</param>
        /// <returns>the new index expression.</returns>
        Expr LoadOffset(IRArray<Expr> indices)
        {
            var offset = IndicesOffset(indices);
            return offset;
        }
    }

    /// <summary>
    /// the data producer.
    /// </summary>
    public interface DataProducer
    {
        /// <summary>
        /// get shape.
        /// </summary>
        /// <returns> shapes. </returns>
        public IRArray<Expr> GetShape();

        /// <summary>
        /// Get the data type of the result.
        /// </summary>
        /// <returns>The data type.</returns>
        public DataType GetDataType();

        /// <summary>
        /// Get the name hint of the data producer.
        /// </summary>
        /// <returns> name string. </returns>
        public string GetNameHint();
    }
}