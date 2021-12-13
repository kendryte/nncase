using Xunit;
using Nncase.TIR;
using Nncase.TIR.Builtin;
using Nncase.IR;
using Nncase.Evaluator;
using System.Collections.Generic;
using TorchSharp;

namespace Nncase.Tests.TIR
{
    public class TBufferTest
    {
        [Fact]
        public void TestBuffer()
        {
            var m = new SizeVar("m");
            var n = new SizeVar("n");
            var l = new SizeVar("l");

            var Ab = TBuffer.Decl((m, n), DataType.Float32);
            var Bb = TBuffer.Decl((n, l), DataType.Float32);

            Assert.IsType<TBuffer>(Ab);
            Assert.Equal(Ab.Dtype, DataType.Float32);
            Assert.Equal(Ab.Shape[0], m);
            Assert.Equal(Ab.Shape[1], n);
        }

        [Fact]
        public void TestBufferAccessPtr()
        {
            var m = new SizeVar("m");
            var n = new SizeVar("n");
            var dict = new Dictionary<Var, torch.Tensor>() {
              { n,  torch.tensor(1) },
              { m,  torch.tensor(3) },
            };
            var Ab = TBuffer.Decl((m, n),
                          DataType.Float32,
                          strides: (n + 1, 1));
            var aptr = Ab.AccessPtr(AccessMode.ReadWrite);
            Assert.Equal(aptr.Parameters[2].Eval(dict), (Ab.Strides[0] * m).Eval(dict));
            Assert.IsType<AccessPtr>(aptr.Target);
        }

        [Fact]
        public void TestBufferAccessPtrOffset()
        {
            var m = new SizeVar("m");
            var n = new SizeVar("n");
            var dict = new Dictionary<Var, torch.Tensor>() {
              { n,  torch.tensor(1) },
              { m,  torch.tensor(3) },
            };
            var Ab = TBuffer.Decl((m, n),
                          DataType.Float32,
                          strides: (n + 1, 1));
            var aptr = Ab.AccessPtr(AccessMode.ReadWrite, offset: 100);
            Assert.Equal(((AccessPtr)aptr.Target).AccessMode, AccessMode.ReadWrite);

            var v = new SizeVar(DType: ElemType.Int32);

            var aptr2 = Ab.AccessPtr(AccessMode.ReadWrite, offset: 100 + 100 + v);

            Testing.AssertExprEqual(aptr2.Parameters[1], 200 + v);
        }

        [Fact]
        public void TestBufferAccessPtrExtent()
        {
            var m = new SizeVar("m");
            var n = new SizeVar("n");
            var Ab = TBuffer.Decl((m, n), DataType.Float32);
            var aptr = Ab.AccessPtr(AccessMode.ReadWrite, offset: 100);
            Testing.AssertExprEqual(aptr.Parameters[2], m * n - 100);
            var Bb = TBuffer.Decl((m, n), DataType.Float32, strides: (n + 1, 1));
            var bptr = Bb.AccessPtr(AccessMode.ReadWrite, offset: 100);
            Testing.AssertExprEqual(bptr.Parameters[2], Bb.Strides[0] * m - 100);
        }

        [Fact]
        public void TestBufferVLoad()
        {
            var m = new SizeVar("m");
            var n = new SizeVar("n");
            var Ab = TBuffer.Decl((m, n), DataType.Float32, elem_offset: 100);
            var load = Ab.VLoad((2, 3));
            Testing.AssertExprEqual(load[Load.Index], 100 + ((2 * n) + 3));
        }
    }
}
