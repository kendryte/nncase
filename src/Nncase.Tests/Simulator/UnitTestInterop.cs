using System;
using System.IO;
using System.Numerics.Tensors;
using System.Runtime.InteropServices;
using Nncase.CodeGen;
using Nncase.IR;
using Nncase.Runtime.Interop;
using Nncase.Schedule;
using Nncase.Simulator;
using Xunit;

namespace Nncase.Tests.SimulatorTest
{
    public class UnitTestInterop
    {
        private readonly byte[] _kmodel;

        public UnitTestInterop()
        {
            var x = new Var("x", new TensorType(DataTypes.Float32, new[] { 1 }));
            var y = x + 1.0f;
            var main = new Function("main", y, new[] { x });
            var module = new IRModule(main);
            var target = CompilerServices.GetTarget("cpu");
            var modelBuilder = new ModelBuilder(target);
            var linkedModel = modelBuilder.Build(module);
            using var output = new MemoryStream();
            linkedModel.Serialize(output);
            _kmodel = output.ToArray();
        }

        [Fact]
        public void TestCreateRTInterpreter()
        {
            var interp = new RTInterpreter();
            Assert.NotNull(interp);
        }

        [Fact]
        public void TestGetHostBufferAllocator()
        {
            var allocator = RTBufferAllocator.Host;
            Assert.NotNull(allocator);
        }

        [Fact]
        public void TestAllocateHostBuffer()
        {
            var allocator = RTBufferAllocator.Host;
            var buffer = allocator.Allocate(256);
            Assert.NotNull(buffer.AsHost());
        }

        [Fact]
        public void TestMapHostBuffer()
        {
            var allocator = RTBufferAllocator.Host;
            var buffer = allocator.Allocate(256).AsHost();
            using (var mmOwner = buffer.Map(RTMapAccess.Write))
            {
                mmOwner.Memory.Span.Fill(1);
            }

            using (var mmOwner = buffer.Map(RTMapAccess.Read))
            {
                Assert.All(mmOwner.Memory.Span.ToArray(), x => Assert.Equal(1, x));
            }
        }

        [Fact]
        public void TestDataTypeCreatePrim()
        {
            var dtype = RTDataType.FromTypeCode(Runtime.TypeCode.Float32);
            Assert.NotNull(dtype);
        }

        [Fact]
        public void TestRTInterpreterLoadModel()
        {
            var interp = new RTInterpreter();
            interp.LoadModel(_kmodel);
            var entry = interp.Entry;
            Assert.NotNull(entry);
            Assert.Equal(1u, entry.ParamsCount);
        }
    }
}
