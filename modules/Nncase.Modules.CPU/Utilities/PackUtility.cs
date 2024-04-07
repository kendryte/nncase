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

    /// <summary>
    /// find the reshape's shape transform matrix.
    /// </summary>
    /// <param name="inShape"> input shape.</param>
    /// <param name="newShape">new shape.</param>
    /// <param name="mat">mat.</param>
    /// <returns>bool.</returns>
    public static bool TryGetShapeMapMatrix(int[] inShape, int[] newShape, out int[,] mat)
    {
        int Dot(int[,] cmat, int i)
        {
            var prod = 1;
            for (int j = 0; j < inShape.Length; j++)
            {
                var v = cmat[i, j] * inShape[j];
                if (v != 0)
                {
                    prod *= v;
                }
            }

            return prod;
        }

        mat = new int[newShape.Length, inShape.Length];
        int i = 0, j = 0;
        var paths = new List<(int, int)>();
        while (i < newShape.Length)
        {
            if (paths.IndexOf((i, j)) != -1)
            {
                return false;
            }

            mat[i, j] = 1;
            paths.Add((i, j));
            var newDim = Dot(mat, i);
            switch (newDim - newShape[i])
            {
                case 0:
                    i++; j++;
                    break;
                case < 0:
                    j++;
                    break;
                case > 0:
                    mat[i, j] = 0;
                    j--;
                    paths.RemoveAt(paths.Count - 1);
                    break;
            }
        }

        return i == newShape.Length && j == inShape.Length;
    }

    /// <summary>
    /// convert the mapping matrix as a dictionary.
    /// the key is in dim, value is not dim.
    /// </summary>
    /// <param name="mat">mat.</param>
    /// <returns>dict.</returns>
    public static (Dictionary<int, List<int>> Forward, Dictionary<int, List<int>> Backward) ShapeMapMatrixAsDict(int[,] mat)
    {
        var forward = new Dictionary<int, List<int>>();
        var backward = new Dictionary<int, List<int>>();
        for (int i = 0; i < mat.GetLength(0); i++)
        {
            for (int j = 0; j < mat.GetLength(1); j++)
            {
                if (mat[i, j] == 0)
                {
                    continue;
                }

                if (!forward.TryGetValue(j, out var l1))
                {
                    l1 = new() { i };
                    forward.Add(j, l1);
                }
                else
                {
                    l1.Add(i);
                }

                if (!backward.TryGetValue(i, out var l2))
                {
                    l2 = new() { j };
                    backward.Add(i, l2);
                }
                else
                {
                    l2.Add(j);
                }
            }
        }

        return (forward, backward);
    }
}
