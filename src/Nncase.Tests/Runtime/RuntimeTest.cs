using System;
using System.Numerics.Tensors;
using System.Runtime.InteropServices;
using Nncase.Runtime;
using Nncase.Schedule;
using Xunit;


namespace Nncase.Tests.RuntimeTest
{
    public class TestRuntimeTensor
    {

        [Fact]
        public void TestCreate()
        {
            var tensor = new DenseTensor<float>(new[] { 1, 2, 3, 4 });
            var rt = RuntimeTensor.FromDense(tensor);
            Assert.Equal(new[] { 1, 2, 3, 4 }, rt.Shape);
            Assert.Equal(new[] { 24, 12, 4, 1 }, rt.Strides);
        }
    }

    public class TestInterpter
    {

        [Fact]
        public void TestCreate()
        {
            var inter = new Interpreter();
            Assert.Throws<InvalidProgramException>(() => new Interpreter());
        }

    }
}