using Xunit;
using System.IO;
using static Nncase.IR.F.Math;
using static Nncase.IR.F.Tensors;
using Nncase.TIR;
using Nncase.IR;



namespace Nncase.Tests.TIRTest
{
    /// <summary>
    /// test the tir construct define
    /// </summary>
    public class ConstructorTest
    {
        string DumpDirPath = Testing.GetDumpDirPath("TIR/ConstructorTest");

        [Fact]
        public void TestExprConstructor()
        {
            var x = Var.Scalar("x", DataType.Float32);
            Assert.IsType<Var>(x);

            var r = new Reduction(
              null, new Expr[] { 1 },
              new[] { new IterVar(TensorType.Scalar(ElemType.Int32), (0, 1), IterMode.CommReduce, 1) },
              null, 0);
            Assert.Null(r.Combiner);
            Assert.Equal(0, r.ValueIndex);

            var lhs = Var.Scalar("x", DataType.Float32) > (Const)1;
            var rhs = Equal(Var.Scalar("x", DataType.Float32), (Const)1);
            var s = new Select(lhs, rhs, (Const)1);
            var buffer_var = Var.Handle("x", DataType.Float32);
            var ld = TIR.T.Load(buffer_var, 1);
            Assert.Equal(ld[Load.Handle], buffer_var);

            var ramp = TIR.T.Ramp(1, 2, 3);
            Assert.Equal((Const)1, ramp[Ramp.Offset]);
            Assert.Equal((Const)2, ramp[Ramp.Stride]);


            var bc = new Broadcast(1000, 10);
            Assert.Equal((Const)1000, bc.Value);

            var sf = new Shuffle(new Expr[] { x }, new Expr[] { 2 });
            Assert.Equal(sf.Vectors[0], x);
            Assert.Equal((Const)2, sf.Indices[0]);

            var lt = new Let(x, 10.0f, x);
            Assert.Equal(lt.Var, x);
            Assert.Equal((Const)10.0f, lt.Value);
            Assert.Equal(lt.Body, x);
        }

        [Fact]
        public void TestBlockConstructor()
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
            func.InferenceType();
            var dumpPath = Path.Combine(DumpDirPath, "TestBlockConstructor");
            func.DumpAsScript("pre", dumpPath);
        }
    }
}