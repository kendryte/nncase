// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

namespace Nncase.Utilities;

public static class IRUtility
{
    /// <summary>
    /// find the reshape's shape transform matrix.
    /// </summary>
    /// <param name="inShape"> input shape.</param>
    /// <param name="newShape">new shape.</param>
    /// <param name="mat">mat [new shape dim, old shpe dim].</param>
    /// <returns>bool.</returns>
    public static bool TryGetShapeMapMatrix(long[] inShape, long[] newShape, out int[,] mat)
    {
        long ProdIn(int[,] cmat, int i)
        {
            long prod = 1;
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

        long ProdOut(int[,] cmat, int j)
        {
            long prod = 1;
            for (int i = 0; i < newShape.Length; i++)
            {
                var v = cmat[i, j] * newShape[i];
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
        while (i >= 0 && i < newShape.Length && j >= 0 && j < inShape.Length)
        {
            if (paths.IndexOf((i, j)) != -1)
            {
                return false;
            }

            mat[i, j] = 1;
            paths.Add((i, j));
            var inDim = ProdIn(mat, i);
            var outDim = ProdOut(mat, j);
            switch (inDim - outDim)
            {
                case 0:
                    i++; j++;
                    break;
                case < 0:
                    j++;
                    break;
                case > 0:
                    if (inDim % newShape[i] == 0)
                    {
                        i++;
                    }
                    else
                    {
                        mat[i, j] = 0;
                        j--;
                        paths.RemoveAt(paths.Count - 1);
                    }

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

    /// <summary>
    /// convert the mapping matrix as a complete dictionary.
    /// the key is in dim, value is not dim.
    /// </summary>
    /// <param name="mat">mat.</param>
    /// <returns>dict.</returns>
    public static (Dictionary<int, List<int>> Forward, Dictionary<int, List<int>> Backward) ShapeMapMatrixAsCompleteDict(int[,] mat)
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

                if (Enumerable.Range(0, mat.GetLength(1)).All(otherJ => j == otherJ || mat[i, otherJ] == 0))
                {
                    // j is the only dim that maps to i
                    if (!forward.TryGetValue(j, out var l1))
                    {
                        l1 = new() { i };
                        forward.Add(j, l1);
                    }
                    else
                    {
                        l1.Add(i);
                    }
                }

                if (Enumerable.Range(0, mat.GetLength(0)).All(otherI => i == otherI || mat[otherI, j] == 0))
                {
                    // i is the only dim that maps to j
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
        }

        return (forward, backward);
    }
}
