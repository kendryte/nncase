using Xunit;
using Nncase.IR;
using Nncase.CodeGen;
using System;
using System.Linq;
using System.Collections.Generic;

namespace Nncase.Tests.CodeGenTest
{
    public interface ICodeGenCase
    {
        /// <summary>
        /// get the entry function
        /// </summary>
        /// <returns></returns>
        public Function GetEntry();
        /// <summary>
        /// custom equal compare method
        /// </summary>
        /// <param name="rtmod"></param>
        public void CompareEqual(IRTModule rtmod);
    }

    public class SubCase : ICodeGenCase
    {
        public Function GetEntry()
        {
            var x = new Var("x", TensorType.Scalar(ElemType.Float32));
            var y = new Var("y", TensorType.Scalar(ElemType.Float32));
            var func = new Function(x - y, x, y);
            return func;
        }

        public void CompareEqual(IRTModule rtmod)
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
        public void CompareEqual(IRTModule rtmod)
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

        public Function GetEntry()
        {
            var n = new TIR.SizeVar("n");
            var A = TIR.Buffer.Decl(new(n), DataType.Int32, "A");
            var i = new Var("i", TensorType.Scalar(ElemType.Int32));
            var j = new Var("j", TensorType.Scalar(ElemType.Int32));
            var out_for = new TIR.For(i, 0, n, TIR.ForMode.Serial);
            var in_for = new TIR.For(j, 0, 10, TIR.ForMode.Serial);
            in_for.Body = new TIR.Sequential(A.Store(i, A[i] + j));
            out_for.Body = new TIR.Sequential(
              A.Store(i, A[i] + 1),
              in_for
            );
            return new Function(out_for, A.Handle, n);
        }
    }

}