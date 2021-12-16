using Nncase.IR;
using Nncase.TIR;
using Xunit;
using Nncase.TIR.F;
using static Nncase.IR.F.Math;

namespace Nncase.Tests.TIR
{
    /// <summary>
    /// test the tir construct define
    /// </summary>
    public class ConstructorTest
    {
        [Fact]
        public void TestExprConstructor()
        {
            var x = Var.Scalar("x", DataType.Float32);
            Assert.IsType<Var>(x);

            var r = new Reduction(
              null, new Expr[] { 1 },
              new[] { new IterVar((0, 1), "x", IterMode.CommReduce) },
              null, 0);
            Assert.Equal(r.Combiner, null);
            Assert.Equal(r.ValueIndex, 0);

            var lhs = Var.Scalar("x", DataType.Float32) > (Const)1;
            var rhs = Equal(Var.Scalar("x", DataType.Float32), (Const)1);
            var s = new Select(lhs, rhs, (Const)1);
            var buffer_var = Var.Handle("x", DataType.Float32);
            var ld = TOps.Load(buffer_var, 1, lhs);
            Assert.Equal(ld[Load.BufferHandle], buffer_var);

            var ramp = TOps.Ramp(1, 2, 3);
            Assert.Equal(ramp[Ramp.BaseOffset].ToScalar<int>(), 1);
            Assert.Equal(ramp[Ramp.Stride].ToScalar<int>(), 2);


            var bc = new Broadcast(1000, 10);
            Assert.Equal(bc.Value.ToScalar<int>(), 1000);

            var sf = new Shuffle(new Expr[] { x }, new Expr[] { 2 });
            Assert.Equal(sf.Vectors[0], x);
            Assert.Equal(sf.Indices[0].ToScalar<int>(), 2);

            var lt = new Let(x, 10.0f, x);
            Assert.Equal(lt.Var, x);
            Assert.Equal(lt.Value.ToScalar<float>(), 10.0f);
            Assert.Equal(lt.Body, x);
        }

        [Fact]
        public void TestStmtConstructor()
        {
            // var v = (Var)"v";
            // var buf_var = Var.Handle("buf", DataType.Float32);
            // var nop = new EvalExpr(1);
            // var lt = new LetStmt(v, 1, nop);
            // Assert.Equal(lt.Var, v);
            // Assert.Equal(lt.Value.ToScalar<int>(), 1);
            // Assert.IsType<EvalExpr>(lt.Body);

            // var ttr = new AttrStmt(Equal(v, 1), "xx", 1, nop);
            // Assert.Equal(ttr.Value.ToScalar<int>(), 1);

            // var ast = new AssertStmt(1, "hellow", nop);
            // Assert.Equal(ast.Body, nop);

            // var fr = new For("x", 0, 10, ForMode.Serial, nop);
            // Assert.Equal(fr.Min.ToScalar<int>(), 0);

            // var st = new Store(buf_var, 1, 10, (Const)1);
            // Assert.Equal(st.BufferHandle, buf_var);
            // Assert.Equal(st.Index.ToScalar<int>(), 10);

            // var alc = new Allocate(buf_var, new Expr[] { 1, 2, 3 }, (Const)true, nop);
            // Assert.Equal(alc.BufferVar, buf_var);


            // var ift = new IfThenElse((Const)false, new EvalExpr(11), nop);
            // Assert.Equal(ift.Else, nop);

            // var bf = Buffer.Decl((1, 2, 3));
            // var pf = new Prefetch(bf, new Range[] { });
            // Assert.IsType<Prefetch>(pf);
        }
    }
}