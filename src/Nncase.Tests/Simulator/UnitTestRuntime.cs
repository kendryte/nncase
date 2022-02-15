using System;
using System.Numerics.Tensors;
using System.Runtime.InteropServices;
using Nncase.Schedule;
using Nncase.Simulator;
using Xunit;

namespace Nncase.Tests.SimulatorTest
{
    public class UnitTestRuntime
    {

        [Fact]
        public void TestCreateRuntimeTensor()
        {
            var tensor = new DenseTensor<float>(new[] { 1, 2, 3, 4 });
            var rt = RuntimeTensor.Create(tensor);
            Assert.Equal(new[] { 1, 2, 3, 4 }, rt.Shape);
            Assert.Equal(new[] { 24, 12, 4, 1 }, rt.Strides);
        }

        [Fact]
        public void TestCreateInterpreter()
        {
            var inter = new Interpreter();
            Assert.Throws<InvalidProgramException>(() => new Interpreter());
        }
    }
}