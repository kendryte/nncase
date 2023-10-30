// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Text;
using System.Threading.Tasks;
using Nncase.IR;
using Nncase.IR.F;
using Nncase.IR.Math;
using Nncase.IR.NN;
using Nncase.Passes;
using Nncase.Passes.Rules.Neutral;
using Nncase.Tests.TestFixture;
using Xunit;
using Math = Nncase.IR.F.Math;
using Random = Nncase.IR.F.Random;

namespace Nncase.Tests.Rules.NeutralTest;

[AutoSetupTestMethod(InitSession = true)]
public class UnitTestSimplifyBinary : TransformTestBase
{
    public static IEnumerable<object[]> TestReassociateMulPositiveData =>
        new[]
        {
            new object[] { new[] { 3 } },
        }.Select((o, i) => o.Concat(new object[] { i }).ToArray());

    public static IEnumerable<object[]> TestReassociateDivPositiveData =>
        new[]
        {
            new object[] { new[] { 3 } },
        }.Select((o, i) => o.Concat(new object[] { i }).ToArray());

    public static IEnumerable<object[]> TestXDivXPositiveData =>
        new[]
        {
            new object[] { new[] { 3 } },
        }.Select((o, i) => o.Concat(new object[] { i }).ToArray());

    public static IEnumerable<object[]> TestCommutateMulPositiveData =>
        new[]
        {
            new object[] { new[] { 3 } },
        }.Select((o, i) => o.Concat(new object[] { i }).ToArray());

    [Theory]
    [MemberData(nameof(TestReassociateMulPositiveData))]
    public void TestReassociateMulPositive(int[] aShape, int index)
    {
        var a = new Var();
        var b = new Var();
        var c = new Var();
        var normal = new Dictionary<Var, IValue>();
        normal.Add(a, Random.Normal(DataTypes.Float32, 0, 1, 0, aShape).Evaluate());
        normal.Add(b, Random.Normal(DataTypes.Float32, 0, 1, 0, aShape).Evaluate());
        normal.Add(c, Random.Normal(DataTypes.Float32, 0, 1, 0, aShape).Evaluate());

        var rootPre = a * b * c; // Math.Binary(binaryOp, Math.Binary(binaryOp, a, bValue), bValue);
        TestMatched<ReassociateMul>(rootPre, normal);
    }

    [Theory]
    [MemberData(nameof(TestReassociateDivPositiveData))]
    public void TestReassociateDivPositive(int[] aShape, int index)
    {
        var a = new Var();
        var b = new Var();
        var c = Random.Normal(DataTypes.Float32, 0, 1, 0, aShape); // Can't get Var's datatype. Pattern will not pass
        var normal = new Dictionary<Var, IValue>();
        normal.Add(a, Random.Normal(DataTypes.Float32, 0, 1, 0, aShape).Evaluate());
        normal.Add(b, Random.Normal(DataTypes.Float32, 0, 1, 0, aShape).Evaluate());

        var rootPre = a * b / c;
        TestMatched<ReassociateDiv>(rootPre, normal);
    }

    [Theory]
    [MemberData(nameof(TestXDivXPositiveData))]
    public void TestXDivXPositive(int[] aShape, int index)
    {
        var a = new Var(new TensorType(DataTypes.Float32, aShape));
        var normal = new Dictionary<Var, IValue>();
        normal.Add(a, Random.Normal(DataTypes.Float32, 0, 1, 0, aShape).Evaluate());

        var rootPre = a / a;
        TestMatched<XDivX>(rootPre, normal);
    }

    [Theory]
    [MemberData(nameof(TestCommutateMulPositiveData))]
    public void TestCommutateMulPositive(int[] aShape, int index)
    {
        var a = new Var();
        var b = new Var();
        var normal = new Dictionary<Var, IValue>();
        normal.Add(a, Random.Normal(DataTypes.Float32, 0, 1, 0, aShape).Evaluate());
        normal.Add(b, Random.Normal(DataTypes.Float32, 0, 1, 0, aShape).Evaluate());
        var rootPre = a * b;
        TestMatched<CommutateMul>(rootPre, normal);
    }
}
