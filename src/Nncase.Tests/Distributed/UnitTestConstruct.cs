// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System.Collections.Generic;
using System.Linq;
using System.Reflection.Metadata;
using NetFabric.Hyperlinq;
using Nncase.Diagnostics;
using Nncase.IR;
using Nncase.IR.Math;
using Nncase.Passes;
using Nncase.Utilities;
using Xunit;

namespace Nncase.Tests.DistributedTest;

[TestFixture.AutoSetupTestMethod(InitSession = true)]
public sealed class UnitTestConstruct : TestClassBase
{
    [Fact]
    public void TestCreate()
    {
        Placement placement = new(Placement.DeviceKind.CPU, new int[] { 2, 2 }, "bt");
        IRArray<SBP> sbps = new[] { SBP.S(2), SBP.S(1), };
        DistributedType t = new(new TensorType(DataTypes.Float32, new[] { 1, 2, 8 }), sbps, placement);
        var input = new Var("input", t);
        var output = IR.F.Math.Unary(UnaryOp.Neg, input);
        output.CheckedType = t;
        CompilerServices.DumpDotIR(output, "create", Dumpper.Directory, false);
    }

    public IReadOnlyList<CallableType> GetCandidateSignatures(Unary unary, TensorType tensorType, Placement placement)
    {
        var layers = new List<List<(SBP, SBP)>>();
        for (int i = 0; i < placement.Rank; i++)
        {
            var layer = new List<(SBP, SBP)>();
            for (int axis = 0; axis < tensorType.Shape.Rank; axis++)
            {
                if (tensorType.Shape[axis] is { IsFixed: true, Value: int s } && s > placement.Hierarchy[i])
                {
                    layer.Add((SBP.S(axis), SBP.S(axis)));
                }
            }

            layer.Add((SBP.B, SBP.B));
            layers.Add(layer);
        }

        return LinqExtensions.CartesianProduct(layers).Select(layer => layer.ToArray()).Select(layer =>
        {
            var inType = new DistributedType(tensorType, Enumerable.Range(0, layers.Count).Select(i => layer[i].Item1).ToArray(), placement);
            var outType = new DistributedType(tensorType, Enumerable.Range(0, layers.Count).Select(i => layer[i].Item2).ToArray(), placement);
            return new CallableType(outType, new[] { inType });
        }).ToList();
    }

    [Fact]
    public void TestGetLeafCandidateNDSBPs()
    {
        Placement placement = new(Placement.DeviceKind.CPU, new int[] { 4, 2 }, "bt");
        var ttype = new TensorType(DataTypes.Float32, new[] { 1, 384, 8192 });
        var input = new Var(ttype);
        var set = new HashSet<DistributedType>(DistributedUtility.GetLeafCandidateNDSBPs(ttype, placement).Select(ndsbp => new DistributedType(ttype, ndsbp, placement)));
        Assert.Equal(9, set.Count);
    }

    [Fact]
    public void TestDistributeTypeEqual()
    {
        Placement placement = new(Placement.DeviceKind.CPU, new int[] { 4, 2 }, "bt");
        var ttype = new TensorType(DataTypes.Float32, new[] { 1, 384, 8192 });
        var a = new DistributedType(ttype, new[] { SBP.S(1), SBP.S(2) }, placement);
        var b = new DistributedType(ttype, new[] { SBP.S(1), SBP.S(2) }, placement);
        Assert.Equal(a, b);
        Assert.StrictEqual(a, b);
    }

    [Fact]
    public void TestUnary()
    {
        Placement placement = new(Placement.DeviceKind.CPU, new int[] { 4, 2 }, "bt");
        var ttype = new TensorType(DataTypes.Float32, new[] { 1, 384, 8192 });
        var input = new Var(ttype);
        var output = IR.F.Math.Unary(UnaryOp.Neg, input);

        var candidates = GetCandidateSignatures((Unary)output.Target, ttype, placement);
        int i = 0;
        foreach (var c in candidates)
        {
            input.CheckedType = c.Parameters[0];
            output.CheckedType = c.ReturnType;
            CompilerServices.DumpDotIR(output, $"unary_{i++}", Dumpper.Directory, false);
        }
    }
}
