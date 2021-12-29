using Xunit;
using System.Linq;
using System.IO;
using System.Collections.Generic;
using System;
using Nncase.IR;
using Nncase.TIR;
using Nncase.CodeGen;
using Nncase.Transform;

namespace Nncase.Tests.CodeGenTest
{

    public class CSourceHostTest
    {
        Target _target = Target.CSourceHost();

        static IEnumerable<object[]> Data =>
          new List<object[]>
          {
              new object[] { new BlockCase() },
              new object[] { new ForCase() },
              new object[] { new SubCase() },
          };

        [Theory]
        [MemberData(nameof(DataOne))]
        public void RunOne(ICodeGenCase Case) => RunCore(Case);

        protected void RunCore(ICodeGenCase Case)
        {
            var dumpDirPath = Testing.GetDumpDirPath($"CodeGenTest/CSourceHostTest/{Case.GetType().Name}");
            var opt = new RunPassOptions(null, 2, dumpDirPath);
            // 1. get function

            var entry = Case.GetEntry();
            var inferResult = entry.InferenceType();
            entry.DumpAsScript("pre", dumpDirPath);
            Assert.True(inferResult);

            // 2. run passes
            var mod = new Module(entry);
            var pmr = new PassManager(mod, opt);
            pmr.Add(Case.GetPass());
            pmr.Run();

            // 3. build re module and compare the function call
            var rtmod = mod.Build(_target);
            rtmod.DumpSource("code", dumpDirPath);
            rtmod.Compile();
            Case.CompareEqual(rtmod);
        }

        // [Theory]
        // [MemberData(nameof(DataAll))]
        // public void RunAll(ICodeGenCase Case) => RunCore(Case);

        public static IEnumerable<object[]> DataOne => Data.Take(1);
        public static IEnumerable<object[]> DataAll => Data.Skip(1);



        [Fact]
        public void TestAdd()
        {
            var x = new Var("x", TensorType.Scalar(ElemType.Float32));
            var y = new Var("y", TensorType.Scalar(ElemType.Float32));
            var func = new Function(x + y, x, y);
            var mod = new Module(func);
            var rtmod = mod.Build(_target);
            Console.WriteLine(rtmod.SourceText);
            rtmod.Compile();
            Assert.Equal(3.5f, rtmod.Invoke(1.2f, 2.3f));
        }
    }
}