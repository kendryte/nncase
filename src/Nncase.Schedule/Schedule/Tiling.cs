// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Google.OrTools.Sat;

namespace Nncase.Schedule;

public static class Tiling
{
    public static void AutoTile()
    {
        var solver = new TilingSolver();
        solver.Solve();
    }
}

internal class TilingSolver
{
    private readonly CpModel _model = new();

    public TilingSolver()
    {
    }

    public void Solve()
    {
        // constants
        const int M = 16;
        const int N = 256;
        const int K = 256;
        const int LoopsCount = 3; // m, n, k
        const int InputOperands = 2;

        const int L2_SIZE = 1024 * 1024 * 4; // 4MB
        const int LDST_PRIM = 128; // 128B
        const int L3_BANDWIDTH = 128; // 128B/cycle
        _ = new CpModel();

        // variables
        _ = CreateTileVars(new[] { M, N, K });
        _ = CreateLoopOrderVars(LoopsCount);
    }

    private IntVar[] CreateTileVars(int[] upperbounds)
    {
        var tiles = new IntVar[upperbounds.Length];
        for (int i = 0; i < tiles.Length; i++)
        {
            tiles[i] = _model.NewIntVar(1, upperbounds[i], $"t{i}");
        }

        return tiles;
    }

    private BoolVar[,] CreateLoopOrderVars(int loopsCount)
    {
        var orders = new BoolVar[loopsCount, loopsCount];
        for (int i = 0; i < loopsCount; i++)
        {
            for (int j = 0; j < loopsCount; j++)
            {
                orders[i, j] = _model.NewBoolVar($"order_d{i}_{j}");
            }
        }

        return orders;
    }

    private BoolVar[,] CreateInputPlaceVars(int inputsCount, int loopsCount)
    {
        throw new NotImplementedException();
    }
}
