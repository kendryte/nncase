// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.IO;
using System.Linq;
using Nncase.IR;
using Nncase.Passes;
using Nncase.TIR;
using Xunit;
using static Nncase.IR.F.Math;
using static Nncase.IR.F.Tensors;

namespace Nncase.Tests.TransformTest;

public sealed class UnitTestSubstitutor : TestClassBase
{
    /// <summary>
    /// the substitutor can't change the inner function var.
    /// </summary>
    [Fact]
    public void TestSubstitutorFailed()
    {
        var loop_i = new Var("loop_i", TensorType.Scalar(DataTypes.Int32));
        T.CreateBuffer(new(DataTypes.Float32, new[] { 1, 2, 3, 4 }), MemoryLocation.Input, out var hd);
        var prim_func_1 = T.PrimFunc("prim_func_1", "k?", T.CreateBuffer(new(DataTypes.Float32, new[] { 1, 2, 3, 4 }), MemoryLocation.Input, out var input_a), T.CreateBuffer(new(DataTypes.Float32, new[] { 1, 2, 3, 4 }), MemoryLocation.Output, out var input_b)).Body(
          T.Load(hd, loop_i)).Build();

        var prim_wrapper = new PrimFunctionWrapper(prim_func_1, 1);

        var input = new Var("input", new TensorType(DataTypes.Float32, new[] { 1, 2, 3, 4 }));
        var main_func = new Function("main", new Call(prim_wrapper, input), input);

        Assert.True(CompilerServices.InferenceType(main_func));

        Dictionary<Expr, Expr> vmap = new() { { loop_i, 1 } };
        var substitutor = Mutator.Substitute(e => vmap.TryGetValue(e, out var res) ? res : null)();

        var main_func_2 = substitutor.Rewrite(main_func);
        Assert.True(object.ReferenceEquals(main_func, main_func_2));
    }

    /// <summary>
    /// Substitute the prim func var.
    /// </summary>
    [Fact]
    public void TestSubstitutorTrue()
    {
        var loop_i = new Var("loop_i", TensorType.Scalar(DataTypes.Int32));
        T.CreateBuffer(new(DataTypes.Float32, new[] { 1, 2, 3, 4 }), MemoryLocation.Input, out var hd);
        var prim_func_1 = T.PrimFunc("prim_func_1", "k?", T.CreateBuffer(new(DataTypes.Float32, new[] { 1, 2, 3, 4 }), MemoryLocation.Input, out var input_a), T.CreateBuffer(new(DataTypes.Float32, new[] { 1, 2, 3, 4 }), MemoryLocation.Output, out var input_b)).Body(
          T.Load(hd, loop_i)).Build();

        Dictionary<Expr, Expr> vmap = new() { { loop_i, 1 } };
        var substitutor = Mutator.Substitute(e => vmap.TryGetValue(e, out var res) ? res : null)();

        var prim_func_2 = substitutor.Rewrite(prim_func_1);
        Assert.True(prim_func_2 is PrimFunction { Body: Sequential { Count: 1 } sequential } && sequential[0] is Call { Arguments: var parameters } && parameters[1] is TensorConst);
    }

    /// <summary>
    /// visit the stackvm var.
    /// </summary>
    [Fact]
    public void TestSubstitutorTrue2()
    {
        var loop_i = new Var("loop_i", TensorType.Scalar(DataTypes.Int32));
        T.CreateBuffer(new(DataTypes.Float32, new[] { 1, 2, 3, 4 }), MemoryLocation.Input, out var hd);
        var prim_func_1 = T.PrimFunc("prim_func_1", "k?", T.CreateBuffer(new(DataTypes.Float32, new[] { 1, 2, 3, 4 }), MemoryLocation.Input, out var input_a), T.CreateBuffer(new(DataTypes.Int32, new[] { 1, 2, 3, 4 }), MemoryLocation.Output, out var input_b)).Body(
          T.Load(hd, loop_i)).Build();

        var prim_wrapper = new PrimFunctionWrapper(prim_func_1, 1);

        var input = new Var("input", new TensorType(DataTypes.Float32, new[] { 1, 2, 3, 4 }));
        var main_func = new Function("main", new Call(prim_wrapper, input) + loop_i, input);

        Assert.True(CompilerServices.InferenceType(main_func));

        Dictionary<Expr, Expr> vmap = new() { { loop_i, 1 } };
        var substitutor = Mutator.Substitute(e => vmap.TryGetValue(e, out var res) ? res : null)();

        var main_func_2 = substitutor.Rewrite(main_func);
        Assert.True(CompilerServices.InferenceType(main_func_2));

        Assert.True(object.ReferenceEquals(main_func, main_func_2));

        Assert.True(main_func_2 is Function { Body: Call { Target: IR.Math.Binary, Arguments: var binary_param } } &&
                  binary_param[0] is Call { Target: PrimFunctionWrapper wrapper } &&
                  object.Equals(prim_wrapper, wrapper) &&
                   binary_param[1] is TensorConst);
    }

    /// <summary>
    /// try substitute the same function twice.
    /// </summary>
    [Fact]
    public void TestSubstitutorTrue3()
    {
        var input = new Var("input", new TensorType(DataTypes.Float32, new[] { 1, 2, 3, 4 }));
        var loop_i = new Var("loop_i", TensorType.Scalar(DataTypes.Int32));
        var main_func = new Function("main", 3 + loop_i, input);

        Assert.True(CompilerServices.InferenceType(main_func));

        Dictionary<Expr, Expr> vmap = new() { { loop_i, 1 } };
        var substitutor = Mutator.Substitute(e => vmap.TryGetValue(e, out var res) ? res : null)();

        var main_func_2 = substitutor.Rewrite(main_func);
        Assert.True(CompilerServices.InferenceType(main_func_2));

        Assert.True(object.ReferenceEquals(main_func, main_func_2));

        Assert.True(main_func_2 is Function { Body: Call { Target: IR.Math.Binary, Arguments: var binary_param } } &&
                   binary_param[1] is TensorConst tensor && tensor.Value.ToScalar<int>() == 1);
    }
}
