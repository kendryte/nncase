using Xunit;
using System.Linq;
using System.IO;
using System.Collections.Generic;
using static Nncase.IR.F.Tensors;
using static Nncase.IR.F.Math;
using Nncase.TIR;
using Nncase.IR;
using Nncase.Transform;

namespace Nncase.Tests.TIRTest
{
    public class TransformTest
    {
        string DumpDirPath = Testing.GetDumpDirPath("TIR/ScheduleTest");

        private static IEnumerable<object[]> Data =>
          new List<object[]>
          {
            new object[] { new ConvertBlocksToOpaqueCase() }
          };

        [Theory]
        [MemberData(nameof(DataOne))]
        public void RunOne(ITransfromCase Case) => RunCore(Case);

        protected void RunCore(ITransfromCase Case)
        {
            var dumpDirPath = Path.Combine(DumpDirPath, Case.GetType().Name);

            var entry = Case.GetEntry();
            var inferResult = entry.InferenceType();
            entry.DumpAsScript("pre", dumpDirPath);
            Assert.True(inferResult);

            var post_entry = Case.RunPass(entry);
            post_entry.DumpAsScript("post", dumpDirPath);
        }
        public static IEnumerable<object[]> DataOne => Data.Take(1);
    }

    public class ConvertBlocksToOpaqueCase : ITransfromCase
    {
        public Function GetEntry()
        {
            var n = T.SizeVar("n");
            var m = T.SizeVar("m");
            var A = T.DeclBuffer((n, m), name: "A");
            var func = T.PrimFunc("func", A.Handle, n, m).Add(
              T.Serial(out var i, n, out var fi).Add(
                T.Serial(out var j, m, out var fj).Add(
                  T.Block("init").
                  Remap(out var vi, out var vj, (fi, fj), "SS").
                  Init(
                    T.Store(A[vi, vj], 1.0f)
                  ).Add(
                    T.Store(A[vi, vj], Cast(vi + vj, DataType.Float32))
                  )
                )
              ),
              n + m
            );
            return func;
        }

        public Function RunPass(Function function)
        {
            var pass = new Transform.TIRPass.ConvertBlocksToOpaquePass();
            return pass.Run(function, RunPassOptions.Invalid);
        }
    }



    public interface ITransfromCase
    {
        public Function GetEntry();

        public Function RunPass(Function function);
    }
}