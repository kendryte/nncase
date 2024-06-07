// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Nncase.IR;
using OrtKISharp;

namespace Nncase.Evaluator;

public static class CPUEvaluatorUtility
{
    public static OrtKISharp.Tensor UnpackTensor(OrtKISharp.Tensor input, IRArray<int> packedAxes, IRArray<int> padNums, out IRArray<int> lanes)
    {
        lanes = input.Shape.TakeLast(packedAxes.Count).Select(i => (int)i).ToArray();
        OrtKISharp.Tensor unpacked = input;
        foreach (var axis in packedAxes.Reverse())
        {
            unpacked = unpacked.Unpack(axis);
        }

        var shape = unpacked.Shape.ToArray();

        OrtKISharp.Tensor sliced = unpacked;
        if (padNums.Any(i => i > 0))
        {
            sliced = OrtKI.Slice(unpacked, Enumerable.Repeat(0L, padNums.Count).ToArray(), Enumerable.Range(0, padNums.Count).Select(i => shape[packedAxes[i]] - padNums[i]).ToArray(), packedAxes.Select(i => (long)i).ToArray(), Enumerable.Range(0, padNums.Count).Select(i => 1L).ToArray());
        }

        return sliced;
    }

    public static OrtKISharp.Tensor RepackTensor(OrtKISharp.Tensor input, IRArray<int> lanes, IRArray<int> packedAxes, IRArray<int> padNums)
    {
        OrtKISharp.Tensor paded = input;
        var shape = input.Shape;

        if (padNums.Any(i => i > 0))
        {
            var pads = Enumerable.Repeat(0L, shape.Length * 2).ToArray();
            for (int i = 0; i < packedAxes.Count; i++)
            {
                pads[shape.Length + packedAxes[i]] = padNums[i];
            }

            // bottom_0,bottom_1,..., top_0, top_1, ...
            paded = OrtKI.Pad(paded, pads, 0f, "constant");
        }

        OrtKISharp.Tensor packed = paded;
        foreach (var (lane, axis) in lanes.Zip(packedAxes))
        {
            packed = packed.Pack(lane, axis);
        }

        return packed;
    }
}
