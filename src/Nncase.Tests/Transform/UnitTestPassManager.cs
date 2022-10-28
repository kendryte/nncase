using System;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.IO;
using System.Linq;
using Nncase.IR;
using Nncase.TIR;
using Nncase.Transform;
using Xunit;
using static Nncase.IR.F.Math;
using static Nncase.IR.F.Tensors;

namespace Nncase.Tests.TransformTest;

public sealed class UnitTestPassManager : TestFixture.UnitTestFixtrue
{
    [Fact]
    public void TestPassMangerUpdateDependence()
    {
        var passOptions = GetPassOptions();

        Dictionary<BaseFunction, BaseFunction> functions_update = new(ReferenceEqualityComparer.Instance);


        var prim_func_1 = T.PrimFunc("prim_func_1", "k?", T.PhysicalBuffer(DataTypes.Float32, Schedule.MemoryLocation.Input, new[] { 1, 2, 3, 4 }, out var input_a), T.PhysicalBuffer(DataTypes.Float32, Schedule.MemoryLocation.Output, new[] { 1, 2, 3, 4 }, out var input_b)).Body(
          T.Nop()
        ).Build();

        var prim_wrapper = new PrimFunctionWrapper(prim_func_1, 1);

        var input = new Var("input", new TensorType(DataTypes.Float32, new[] { 1, 2, 3, 4 }));
        var main_func = new Function("main", new Call(prim_wrapper, ImmutableArray.Create<Expr>(input)), ImmutableArray.Create<Var>(input));

        // prim_func_2 for update
        var prim_func_2 = T.PrimFunc("prim_func_2", "k?", T.PhysicalBuffer(DataTypes.Float32, Schedule.MemoryLocation.Input, new[] { 1, 2, 3, 4 }, out var input_a_2), T.PhysicalBuffer(DataTypes.Float32, Schedule.MemoryLocation.Output, new[] { 1, 2, 3, 4 }, out var input_b_2)).Body(
          T.Nop(),
          T.Nop()
        ).Build();

        Assert.True(CompilerServices.InferenceType(main_func));
        Assert.True(CompilerServices.InferenceType(prim_func_2));

        functions_update.Add(prim_func_1, prim_func_2);


        var module = new IR.IRModule(main_func);
        module.Add(prim_wrapper);
        module.Add(prim_func_1);

        var mutator = new DependenceMutator(functions_update);
        var post = mutator.Visit(module.Entry!);

        CompilerServices.DumpIR(module.Entry, "pre", passOptions.DumpDir);
        CompilerServices.DumpIR(post, "post", passOptions.DumpDir);
        Assert.True(post is Function { Body: Call { Target: PrimFunctionWrapper { Target: PrimFunction { Name: "prim_func_2" } } } });
    }


}
