using Nncase.IR;
using Nncase.TIR;
using Xunit;
using Nncase.TIR.F;
using static Nncase.IR.F.Math;
using System;

namespace Nncase.Tests.TIRTest
{
    /// <summary>
    /// test the tir construct define
    /// </summary>
    public class ModuleTest
    {

        [Fact]
        public void TestName()
        {
            var lhs = new Var("lhs", DataType.Float32);
            var rhs = new Var("rhs", DataType.Float32);
            var output = lhs + rhs;
            var func = new Function(output, lhs, rhs);
            func.InferenceType();
            Console.Write(func.DumpExprAsIL());
            var mod = new IRModule();
            mod.Add(func);
            mod.Entry = func;
        }
    }
}
