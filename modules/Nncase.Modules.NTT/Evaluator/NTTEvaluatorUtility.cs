// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Nncase.IR;
using OrtKISharp;

namespace Nncase.Evaluator;

public static class NTTEvaluatorUtility
{
    public static OrtKISharp.Tensor DevectorizeTensor(OrtKISharp.Tensor input, IRArray<int> vectorizedAxes, IRArray<int> padNums, out IRArray<int> lanes)
    {
        lanes = input.Shape.TakeLast(vectorizedAxes.Count).Select(i => (int)i).ToArray();
        OrtKISharp.Tensor devectorized = input;
        devectorized = devectorized.Unpack(vectorizedAxes.Count, vectorizedAxes);
        var shape = devectorized.Shape.ToArray();

        OrtKISharp.Tensor sliced = devectorized;
        if (padNums.Any(i => i > 0))
        {
            sliced = OrtKI.Slice(devectorized, Enumerable.Repeat(0L, padNums.Count).ToArray(), Enumerable.Range(0, padNums.Count).Select(i => shape[vectorizedAxes[i]] - padNums[i]).ToArray(), vectorizedAxes.Select(i => (long)i).ToArray(), Enumerable.Range(0, padNums.Count).Select(i => 1L).ToArray());
        }

        return sliced;
    }

    public static OrtKISharp.Tensor RevectorizeTensor(OrtKISharp.Tensor input, IRArray<int> lanes, IRArray<int> vectorizedAxes, IRArray<int> padNums)
    {
        OrtKISharp.Tensor paded = input;
        var shape = input.Shape;

        if (padNums.Any(i => i > 0))
        {
            var pads = Enumerable.Repeat(0L, shape.Length * 2).ToArray();
            for (int i = 0; i < vectorizedAxes.Count; i++)
            {
                pads[shape.Length + vectorizedAxes[i]] = padNums[i];
            }

            // bottom_0,bottom_1,..., top_0, top_1, ...
            paded = OrtKI.Pad(paded, pads, 0f, "constant");
        }

        OrtKISharp.Tensor vectorized = paded;
        vectorized = vectorized.Pack(0, lanes, vectorizedAxes);
        return vectorized;
    }
}
