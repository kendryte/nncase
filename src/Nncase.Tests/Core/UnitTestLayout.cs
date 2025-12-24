// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System.Linq;
using Nncase.IR;
using Nncase.Layouts;
using Xunit;
using Lutil = Nncase.Layouts.LayoutUtilities;

namespace Nncase.Tests.CoreTest;

public sealed class UnitTestLayout
{
    [Fact]
    public void TestMakeLayout()
    {
        var layout = new Layout([2048, 2048], [2048, 1]);
        System.Console.WriteLine(layout);
    }

    [Fact]
    public void TestZippedDivide()
    {
        var layout = new Layout([2048, 2048], [2048, 1]);
        var newLayout = Lutil.ZippedDivide(layout, 4);
        System.Console.WriteLine(newLayout);
        Assert.Equal("Layout((4, (512, 2048)):(2048, (8192, 1)))", newLayout.ToString());
    }

    [Fact]
    public void TestCoordOffset()
    {
        var layout = new Layout([2048, 2048], [2048, 1]);
        var newLayout = Lutil.TiledDivide(layout[0], 1024);
        newLayout = Lutil.TiledDivide(newLayout, new CollectValue([256]));
        System.Console.WriteLine(newLayout);
        Assert.Equal("Layout(((256,), 4, 2):((2048,), 524288, 2097152))", newLayout.ToString());
    }

    [Fact]
    public void TestUnflatten()
    {
        var shape = new CollectValue([2048]);
        var unflattened = Lutil.Unflatten(shape, 1);
        Assert.Equal("2048", unflattened.ToString());
        unflattened = Lutil.Unflatten(2048, new CollectValue([1]));
        Assert.Equal("(2048,)", unflattened.ToString());
        Assert.Throws<System.ArgumentException>(() =>
        {
            Lutil.Unflatten(new CollectValue([2048, 128]), new CollectValue([1]));
        });
    }

    [Fact]
    public void TestDistributedTypeLayout()
    {
        var placement = new Placement([1, 2, 8, 4, 4], "cdyxt");
        var distType = new DistributedType(new TensorType(DataTypes.Float32, new[] { 2048, 1024 }), new SBP[] { SBP.S([1, 3]), SBP.B }, placement);
        var layout = Layout.From(distType.TensorType);
        Assert.Equal("Layout((2048, 1024):(1024, 1))", layout.ToString());

        // from the distributed create the tiler.
        var tiler = Lutil.GetTiler(layout.Shape, distType.AxisPolicies, distType.Placement);
        var shard = Lutil.ZippedDivide(layout, tiler);

        // the (2,4) is tiler from the split sharding.
        Assert.Equal("Layout((((2, 4), 1), (256, 1024)):(((1048576, 262144), 0), (1024, 1)))", shard.ToString());
        var filtered = Lutil.Filter(shard);
        Assert.Equal("Layout((2, 4, 256, 1024):(1048576, 262144, 1024, 1))", filtered.ToString());
    }

    [Fact]
    public void TestCoordinate()
    {
        var layout = new Layout([2048, 2048], [2048, 1]);
        CollectValue coord = [1000, 5];
        var offset = Lutil.Crd2Idx(coord, layout.Shape, layout.Stride);
        Assert.Equal((1000 * 2048) + 5, offset);

        layout = new Layout([8, new CollectValue([2048, 128])], [128, new CollectValue([1024, 1])]);
        coord = [3, new CollectValue([0, 0])];
        offset = Lutil.Crd2Idx(coord, layout.Shape, layout.Stride);
        Assert.Equal(3 * 128, offset);
    }

    [Fact]
    public void TestCoorinateWithDifferentLayout()
    {
        var placement = new Placement([1, 2, 8, 4, 4], "cdyxt");
        var tensorType = new TensorType(DataTypes.Float32, new[] { 2048, 1024 });
        var distTypeA = new DistributedType(tensorType, new SBP[] { SBP.S([1, 3]), SBP.B }, placement);
        var distTypeB = new DistributedType(tensorType, new SBP[] { SBP.B, SBP.S([2]) }, placement);
        var shardA = Layout.From(distTypeA);
        var shardB = Layout.From(distTypeB);

        // clac the minimal tilesize.
        var minTile = Lutil.Minimum(shardA[1].Shape, shardB[1].Shape);

        // coordinate in layoutA
        long[] destPid = [0, 1, 3, 2, 1]; // sampled process id.
        long[] GetUsedPid(long[] longs, IRArray<SBP> axisPolicies)
        {
            return axisPolicies.OfType<SBPSplit>().Select(x => x.Axes).SelectMany(x => x).Select(x => longs[x]).ToArray();
        }

        var aPid = GetUsedPid(destPid, distTypeA.AxisPolicies);
        var bPid = GetUsedPid(destPid, distTypeB.AxisPolicies);

        var localLayoutB = new Layout(shardB[1].Shape);
        var localLayoutBTiled = Lutil.ZippedDivide(localLayoutB, minTile); // [(minTile), (times)]
        var minTileTimes = localLayoutBTiled[1];
        for (int i = 0; i < (int)minTileTimes.Size(); i++)
        {
            var localTiledcoord = new CollectValue([RecursiveValue.UnderScore, Lutil.Idx2Crd(i, minTileTimes.Shape)]);
            var localTiledoffset = Lutil.Crd2Idx(localTiledcoord, localLayoutBTiled.Shape, localLayoutBTiled.Stride);
            var localBCoord = Lutil.Idx2Crd(localTiledoffset, localLayoutB.Shape, localLayoutB.Stride); // dest start

            CollectValue shardBCoord = [new CollectValue(bPid.Select(x => new IntValue(x))), localBCoord];
            var shardBoffset = shardB.Invoke(shardBCoord);
            var shardACoord = Lutil.Idx2Crd(shardBoffset, shardA.Shape, shardA.Stride); // src rank with src start

            // System.Console.WriteLine(shardACoord);
        }
    }
}
