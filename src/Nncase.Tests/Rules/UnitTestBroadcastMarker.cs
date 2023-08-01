// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Nncase.IR;
using Nncase.Passes.Rules.Lower;
using Nncase.Tests.TestFixture;
using Xunit;
using static Nncase.IR.F.Math;
using static Nncase.IR.F.Tensors;

namespace Nncase.Tests.Rules;

[AutoSetupTestMethod(InitSession = true)]
public class UnitTestBroadcastMarker : TransformTestBase
{
    [Fact]
    public void TestBroadcastInputMarker()
    {
        var input = Testing.Rand<float>(1, 3, 24, 24);
        var a = Abs(Reshape(new Marker(WellknownMarkerNames.RangeOf, input, new[] { -1f, 1f }), input.Shape));
        var result = TestMatched<BroadcastInputMarker>(a);
        TestNotMatch<BroadcastInputMarker>(result);
    }

    [Fact]
    public void TestBroadcastOutputMarker()
    {
        var input = Testing.Rand<float>(1, 3, 24, 24);
        var a = new Marker(WellknownMarkerNames.RangeOf, Reshape(Abs(input), input.Shape), new[] { -1f, 1f });
        var result = TestMatched<BroadcastOutputMarker>(a);
        TestNotMatch<BroadcastOutputMarker>(result);
    }
}
