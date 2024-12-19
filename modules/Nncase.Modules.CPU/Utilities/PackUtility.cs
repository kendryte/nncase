// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Nncase.IR;

namespace Nncase.Utilities;

public static class PackUtility
{
    public static Expr PadForPack(Expr input, int[] shape, int[] packedAxes, int[] lanes, Expr value, out int[] padNums)
    {
        var isPadded = false;
        var pads = new int[shape.Length, 2];
        for (int i = 0; i < packedAxes.Length; i++)
        {
            var axis = packedAxes[i];
            if (shape[axis] % lanes[i] != 0)
            {
                pads[axis, 1] = MathUtility.AlignUp(shape[axis], lanes[i]) - shape[axis];
                isPadded = true;
            }
        }

        padNums = new int[packedAxes.Length];
        for (int i = 0; i < packedAxes.Length; i++)
        {
            padNums[i] = pads[packedAxes[i], 1];
        }

        if (isPadded)
        {
            return IR.F.NN.Pad(input, pads, PadMode.Constant, value);
        }

        return input;
    }

    public static Expr SliceForPack(Expr input, int[] shape, int[] padNums)
    {
        bool isPadded = false;
        var ends = shape.ToArray();
        if (padNums.Any(i => i > 0))
        {
            isPadded = true;
        }

        return isPadded ? IR.F.Tensors.Slice(input, Enumerable.Repeat(0, shape.Length).ToArray(), ends, shape.Length) : input;
    }
}
