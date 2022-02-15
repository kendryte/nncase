using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Nncase.IR;
using Nncase.TIR;
using Nncase.Transform;
using Xunit;
using static Nncase.IR.F.Math;
using static Nncase.IR.F.Tensors;

namespace Nncase.Tests.TIRTest
{
    public class TransformTest
    {
        string DumpDirPath = Testing.GetDumpDirPath("TIR/TransformTest");

        private static IEnumerable<object[]> Data =>
          new List<object[]>
          {
            new object[] { new ConvertBlocksToOpaqueCase() },
            new object[] { new FlattenBufferCase() },
            new object[] { new LowerBlockInitCase() },
          };

        [Theory]
        [MemberData(nameof(DataOne))]
        public void RunOne(ITransfromCase Case) => RunCore(Case);


        protected void RunCore(ITransfromCase Case)
        {
            var dumpDirPath = Path.Combine(DumpDirPath, Case.GetType().Name);
            var options = new RunPassOptions(null, 2, dumpDirPath);

            var entry = Case.GetEntry();
            var inferResult = entry.InferenceType();
            entry.DumpAsScript("pre", dumpDirPath);
            Assert.True(inferResult);

            var post_entry = Case.Pass.Run(entry, options);
            post_entry.DumpAsScript("post", dumpDirPath);
        }

        [Theory]
        [MemberData(nameof(DataAll))]
        public void RunAll(ITransfromCase Case) => RunCore(Case);

        public static IEnumerable<object[]> DataOne => Data.Take(1);
        public static IEnumerable<object[]> DataAll => Data.Skip(1);
    }

    public class LowerBlockInitCase : ConvertBlocksToOpaqueCase
    {
        public LowerBlockInitCase()
        {
            Pass.Add(
                  new Transform.Mutator.LowerBlockInit()
            );
        }
    }


    public class FlattenBufferCase : ConvertBlocksToOpaqueCase
    {
        public FlattenBufferCase()
        {
            Pass.Add(new Transform.Mutator.LowerBlockInit());
            Pass.Add(new Transform.Mutator.ConvertBlocksToOpaque());
            Pass.Add(new Transform.Mutator.FlattenBuffer());
        }
    }

    public class ConvertBlocksToOpaqueCase : ITransfromCase
    {
        public ConvertBlocksToOpaqueCase()
        {
            Pass.Add(
              new Transform.Mutator.ConvertBlocksToOpaque()
            );
        }
        public override Function GetEntry()
        {
            var n = T.SizeVar("n");
            var m = T.SizeVar("m");
            var A = T.DeclBuffer((n, m), name: "A");
            var func = T.PrimFunc("func", A.Handle, n, m).Body(
              T.Serial(out var i, n, out var fi).Body(
                T.Serial(out var j, m, out var fj).Body(
                  T.Block("init").
                  Remap(out var vi, out var vj, (fi, fj), "SS").
                  Init(
                    T.Store(A[vi, vj], 1.0f)
                  ).Body(
                    T.Store(A[vi, vj], Cast(vi + vj, DataType.Float32))
                  )
                )
              ),
              n + m
            );
            return func;
        }
    }

    public abstract class ITransfromCase
    {
        public TIRPass Pass = new("TIRPass");
        public virtual Function GetEntry()
        {
            throw new NotImplementedException("GetEntry");
        }
    }
}