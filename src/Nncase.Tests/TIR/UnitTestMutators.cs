// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System.IO;
using System.Threading.Tasks;
using Nncase.IR;
using Nncase.Tests.TestFixture;
using Nncase.TIR;
using Nncase.Transform;
using Nncase.Transform.Mutators;
using Xunit;
using static Nncase.IR.F.Math;
using static Nncase.IR.F.Tensors;

namespace Nncase.Tests.TIRTest;

[AutoSetupTestMethod(InitSession = true)]
public sealed class UnitTestMutators : TestClassBase
{
    [Fact]
    public async Task TestUnRollLoopSequential()
    {
        var main = T.PrimFunc("main", Callable.StackVMModuleKind).Body(// (*i8) -> ()
            T.Unrolled(out var i, (0, 32, 4)).Body(// ()
              T.Unrolled(out var j, (0, 16, 4)).Body(// ()
                T.Unrolled(out var k, (0, 18, 6)).Body(// ()
                  T.Unrolled(out var l, (0, 32, 16)).Body(// ()
                    T.Block("block").Body()))))).Build();

        CompilerServices.InferenceType(main);
        var prmg = CompileSession.CreatePassManager("prmg");
        prmg.AddWithName<PrimFuncPass>("unroll").Configure(
          p =>
          {
              p.Add<UnRollLoopSequential>();
          });

        var module = new IR.IRModule(main);
        await prmg.RunAsync(module);
        Assert.Equal(32 / 4 * (16 / 4) * (18 / 6) * (32 / 16), ((TIR.PrimFunction)module.Entry!).Body.Count);
    }
}
