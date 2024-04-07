// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Linq;
using Google.OrTools.Sat;
using Nncase.IR;
using Nncase.Utilities;
using Xunit;

namespace Nncase.Tests.Rules.Packing;

public sealed class PackUtilityTest
{
    [Theory]
    [InlineData(new object[] { new int[] { 1, 3, 2, 3, 1, 1, 7 }, new int[] { 1, 1, 3, 6, 1, 7 }, true })]
    [InlineData(new object[] { new int[] { 2, 3, 4 }, new int[] { 4, 3, 2 }, false })]
    public void TestComputeReshapeMapping(int[] inShape, int[] newShape, bool valid)
    {
        Assert.Equal(valid, PackUtility.TryGetShapeMapMatrix(inShape, newShape, out var mat));
        if (valid)
        {
#if DEBUG
            DisplayMat(mat);
#endif
        }
    }

    [Fact]
    public void TestSolveReshapeMapping()
    {
        var inshape = new int[] { 1, 3, 2, 3, 1, 1, 7 };
        var newshape = new int[] { 1, 1, 3, 6, 1, 7 };

        var model = new CpModel();
        var mat = new BoolVar[newshape.Length, inshape.Length];
        for (int i = 0; i < mat.GetLength(0); i++)
        {
            for (int j = 0; j < mat.GetLength(1); j++)
            {
                mat[i, j] = model.NewBoolVar($"{i}_{j}");
            }
        }

        for (int i = 0; i < mat.GetLength(0); i++)
        {
            model.Add(model.NewConstant(newshape[i]) == LinearExpr.Sum(Enumerable.Range(0, mat.GetLength(1)).Select(j => inshape[j] * mat[i, j])));
        }

        // sum(colum) >= 1
        for (int j = 0; j < mat.GetLength(1); j++)
        {
            model.AddAtLeastOne(Enumerable.Range(0, mat.GetLength(0)).Select(i => mat[i, j]));
        }

        model.Minimize(LinearExpr.Sum(Enumerable.Range(0, mat.GetLength(0)).Select(i => Enumerable.Range(0, mat.GetLength(1)).Select(j => (i, j))).SelectMany(p => p).Select(p => mat[p.i, p.j])));
        var solver = new CpSolver();
        var status = solver.Solve(model);
        if (status is not (CpSolverStatus.Optimal or CpSolverStatus.Feasible))
        {
            System.Console.WriteLine(status);
            return;
        }
    }

    private void DisplayMat(int[,] mat)
    {
        for (int i = 0; i < mat.GetLength(0); i++)
        {
            System.Console.WriteLine(string.Join(", ", Enumerable.Range(0, mat.GetLength(1)).Select(j => mat[i, j].ToString())));
        }
    }
}
