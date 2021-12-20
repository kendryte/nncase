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
        public Function GetEntry();
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

        public void CompareEqual(IRTModule rtmod)
        {
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
            return new Function(out_for, n);
        }
    }

}