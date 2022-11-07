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


    [Fact]
    public void TestPassMangerUpdateDependence2()
    {
        var passOptions = GetPassOptions();

        Dictionary<BaseFunction, BaseFunction> functions_update = new(ReferenceEqualityComparer.Instance);

        /* only update func_1 
          %0 = %func_0(%input): // f32[1,3,24,32]
          %1 = %func_1(%0): // i8[1,3,24,32]
          %3 = %func_3(%2): // f16[1,23,30,16]
        */


        var prim_func_0 = T.PrimFunc("prim_func_0", "k?", T.PhysicalBuffer(DataTypes.Float32, Schedule.MemoryLocation.Input, new[] { 1, 24, 32, 3 }, out var _), T.PhysicalBuffer(DataTypes.Float32, Schedule.MemoryLocation.Output, new[] { 1, 3, 24, 32 }, out var _)).Body(
          T.Nop()
        ).Build();
        var func_0 = new PrimFunctionWrapper(prim_func_0, 1);

        var prim_func_1 = T.PrimFunc("prim_func_1", "k?", T.PhysicalBuffer(DataTypes.Float32, Schedule.MemoryLocation.Input, new[] { 1, 3, 24, 32 }, out var _), T.PhysicalBuffer(DataTypes.Float32, Schedule.MemoryLocation.Output, new[] { 1, 3, 24, 32 }, out var _)).Body(
          T.Nop()
        ).Build();
        var func_1 = new PrimFunctionWrapper(prim_func_1, 1);

        var prim_func_2 = T.PrimFunc("prim_func_2", "k?", T.PhysicalBuffer(DataTypes.Float32, Schedule.MemoryLocation.Input, new[] { 1, 3, 24, 32 }, out var _), T.PhysicalBuffer(DataTypes.Float32, Schedule.MemoryLocation.Output, new[] { 1, 23, 30, 16 }, out var _)).Body(
          T.Nop()
        ).Build();
        var func_2 = new PrimFunctionWrapper(prim_func_2, 1);

        var input = new Var("input", new TensorType(DataTypes.Float32, new[] { 1, 24, 32, 3 }));
        var main_func = new Function("main",
                        new Call(func_2, new Call(func_1, new Call(func_0, ImmutableArray.Create<Expr>(input)))),
                        ImmutableArray.Create<Var>(input));
        Assert.True(CompilerServices.InferenceType(main_func));

        // prim_func_2 for update
        var prim_func_1_update = T.PrimFunc("prim_func_1_update", "k?", T.PhysicalBuffer(DataTypes.Float32, Schedule.MemoryLocation.Input, new[] { 1, 3, 24, 32 }, out var _), T.PhysicalBuffer(DataTypes.Float32, Schedule.MemoryLocation.Output, new[] { 1, 3, 24, 32 }, out var _)).Body(
          T.Nop(),
          T.Nop()
        ).Build();

        Assert.True(CompilerServices.InferenceType(prim_func_1_update));

        functions_update.Add(prim_func_1, prim_func_1_update);


        var module = new IR.IRModule(main_func);
        module.Add(func_0);
        module.Add(prim_func_0);
        module.Add(func_1);
        module.Add(prim_func_1);
        module.Add(func_2);
        module.Add(prim_func_2);

        var mutator = new DependenceMutator(functions_update);
        var post = mutator.Visit(module.Entry!);

        CompilerServices.DumpIR(module.Entry, "pre", passOptions.DumpDir);
        CompilerServices.DumpIR(post, "post", passOptions.DumpDir);

        for (int i = 0; i < module.Functions.Count; i++)
        {
            if (functions_update.TryGetValue(module.Functions[i], out var updated_func))
                module.Update(i, updated_func);
        }

        Assert.True(object.ReferenceEquals(((PrimFunctionWrapper)module.Functions[1]).Target, module.Functions[2]));
        Assert.True(object.ReferenceEquals(((PrimFunctionWrapper)module.Functions[3]).Target, module.Functions[4]));
        Assert.True(object.ReferenceEquals(((PrimFunctionWrapper)module.Functions[5]).Target, module.Functions[6]));
    }
}
