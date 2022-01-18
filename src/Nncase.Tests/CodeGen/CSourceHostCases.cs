using System;
using System.Collections.Generic;
using System.Linq;
using System.Numerics.Tensors;
using Nncase.CodeGen;
using Nncase.IR;
using Nncase.TIR;
using Nncase.Transform;
using Xunit;

namespace Nncase.Tests.CodeGenTest
{

    class EmptyPass : FunctionPass
    {
        public EmptyPass() : base("EmptyPass") { }
        protected override Function RunCore(Function function, RunPassOptions options)
        {
            return function;
        }
    }

    public abstract class ICodeGenCase
    {
        /// <summary>
        /// get the entry function
        /// </summary>
        /// <returns></returns>
        public abstract Function GetEntry();

        /// <summary>
        /// custom equal compare method
        /// </summary>
        /// <param name="rtmod"></param>
        public abstract void CompareEqual(IRTModel rtmod);

        public virtual FunctionPass GetPass()
        {
            return new EmptyPass();
        }
    }

    public class SubCase : ICodeGenCase
    {
        public override Function GetEntry()
        {
            var x = new Var("x", TensorType.Scalar(ElemType.Float32));
            var y = new Var("y", TensorType.Scalar(ElemType.Float32));
            var func = new Function(x - y, x, y);
            return func;
        }

        public override void CompareEqual(IRTModel rtmod)
        {
            Assert.Equal(2.3f - 2.1f, rtmod.Invoke(2.3f, 2.1f));
        }
    }

    public class ForCase : ICodeGenCase
    {
        void RefFunc(int[] A, int n)
        {
            for (int i = 0; i < n; i++)
            {
                A[i] = A[i] + 1;
                for (int j = 0; j < 10; j++)
                {
                    A[i] = A[i] + j;
                }
            }
        }

        /// <inheritdoc/>
        public override void CompareEqual(IRTModel rtmod)
        {
            var rand = new Random();
            int n = 12;
            var A1 = Enumerable.Range(0, n).Select(i => rand.Next(456)).ToArray();
            var A2 = new int[n];
            A1.CopyTo(A2, 0);

            RefFunc(A1, n);
            rtmod.Invoke(A2, n);
            Assert.True(Enumerable.Range(0, n).All(i => A1[i] == A2[i]));
        }

        public override Function GetEntry()
        {
            var n = T.SizeVar("n");
            var A = TIR.T.DeclBuffer(new(n), DataType.Int32, "A");
            var out_for =
             T.Serial(out var i, n, out _).Body(
              T.Store(A[i], A[i] + 1),
              T.Serial(out var j, n, out _).Body(
                T.Store(A[i], A[i] + j)
              )
            );
            return T.PrimFunc("main", A.Handle, n).Body(out_for);
        }
    }

    public class ForGridCase : ICodeGenCase
    {

        void RefFunc(int[] A, int n, int m)
        {
            for (int i = 0; i < n; i++)
            {
                for (int j = 0; j < m; j++)
                {
                    A[i * n + j] = i + j;
                }
            }
        }

        public override void CompareEqual(IRTModel rtmod)
        {
            int n = 10, m = 20;
            var A1 = new DenseTensor<int>(new[] { n, m }).ToArray();
            var A2 = new DenseTensor<int>(new[] { n, m }).ToArray();
            RefFunc(A1, n, m);
            rtmod.Invoke(A2, n, m);
            Assert.True(Enumerable.Range(0, n * m).All(i => A1[i] == A2[i]));
        }

        public override Function GetEntry()
        {
            var n = T.SizeVar("n");
            var m = T.SizeVar("m");
            var A = TIR.T.DeclBuffer((n, m), DataType.Int32, "A");
            var func = T.PrimFunc("main", A.Handle, n, m).Body(
              T.Grid(out var i, out var j, (n, m)).Body(
                T.Store(A[i * n + j], i + j)
              )
            );
            return func;
        }
    }

    public class BlockCase : ICodeGenCase
    {

        public override Function GetEntry()
        {
            var n = T.SizeVar("n");
            var m = T.SizeVar("m");
            var A = T.DeclBuffer((n, m), name: "A");
            var func = T.PrimFunc("func", A.Handle, n, m).Body(
            T.Grid(out var i, out var j, (n, m), out var lp).Body(
              T.Block("init").Remap(out var vi, out var vj, (lp.i, lp.j), "SS").
              Init(
                T.Store(A[vi, vj], 1.0f)
              ).Body(
                T.Store(A[vi, vj], IR.F.Tensors.Cast(vi + vj, DataType.Float32))
              )
            ),
            n + m
            );
            return func;
        }

        public int RefFunc(float[] A, int n, int m)
        {
            for (int i = 0; i < n; i++)
            {
                for (int j = 0; j < m; j++)
                {
                    A[i * m + j] = i + j;
                }
            }
            return n + m;
        }

        public override void CompareEqual(IRTModel rtmod)
        {
            int n = 10, m = 12;
            var A1 = new float[n * m];
            var A2 = new float[n * m];
            var r1 = RefFunc(A1, n, m);
            var r2 = rtmod.Invoke(A1, n, m);
            Assert.Equal(r1, r2);
        }

        public override FunctionPass GetPass()
        {
            var pass = new TIRPass("TIRPass");
            pass.Add(
                new Transform.Mutator.LowerBlockInit(),
                new Transform.Mutator.ConvertBlocksToOpaque(),
                new Transform.Mutator.FlattenBuffer()
            );
            return pass;
        }
    }
}