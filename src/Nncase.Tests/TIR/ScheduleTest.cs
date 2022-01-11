using Xunit;
using System;
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
    public class ScheduleTest
    {
        string DumpDirPath = Testing.GetDumpDirPath("TIR/ScheduleTest");

        private static IEnumerable<object[]> Data =>
          new List<object[]>
          {
            new object[] { new SplitCase() },
          };

        [Theory]
        [MemberData(nameof(DataOne))]
        public void RunOne(IScheduleCase Case) => RunCore(Case);

        protected void RunCore(IScheduleCase Case)
        {
            var dumpDirPath = Path.Combine(DumpDirPath, Case.GetType().Name);
            var options = new RunPassOptions(null, 2, dumpDirPath);

            var entry = Case.GetEntry();
            var inferResult = entry.InferenceType();
            entry.DumpAsScript("pre", dumpDirPath);
            Assert.True(inferResult);
            
            var sch = new Scheduler(entry);
            Case.Schedule(sch);
            sch.Entry.InferenceType();
            sch.Entry.DumpAsScript("post", dumpDirPath);
        }
        public static IEnumerable<object[]> DataOne => Data.Take(1);
    }

    public class SplitCase : IScheduleCase
    {
        public override Function GetEntry()
        {
            var m = T.SizeVar("m");
            var A = T.DeclBuffer((12, m), name: "A");
            var func = T.PrimFunc("func", A.Handle, m).Add(
              T.Serial(out var i, (0, 12), out var fi).Add(
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
              m
            );
            return func;
        }

        public override void Schedule(Scheduler sch)
        {
            var b = sch.GetBlock("init");
            var loops = sch.GetLoops(b);
            var lvs = sch.Split(loops[0], 3, 4);
        }
    }

    public abstract class IScheduleCase
    {
        public abstract Function GetEntry();
        public abstract void Schedule(Scheduler scheduler);
    }
}