// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.Diagnostics;
using System.Linq;
using System.Numerics;
using System.Text;
using System.Threading.Tasks;
using Google.OrTools.ConstraintSolver;
using Nncase.Utilities;

namespace Nncase.Schedule;

public static class Tiling
{
    public static void AutoTile()
    {
        var solver = new TilingSolver(new[] { 16, 256, 256 }, 3, new LoopMask[] { new(0b101), new(0b110), new(0b011) });
        solver.Solve();
    }
}

public struct LoopMask
{
    private readonly uint _mask;

    public LoopMask(uint mask)
    {
        _mask = mask;
    }

    public int Count => BitOperations.PopCount(_mask);

    public bool IsRelated(int loop) => (_mask & (1 << loop)) != 0;
}

public record struct OrderCombination(LoopMask Loops)
{
    public IntExpr Expr { get; set; }
}

public class TilingSolver
{
    // 1. Constants
    private const int L2_SIZE = 1024 * 1024 * 4; // 4MB
    private const int L3_BANDWIDTH = 128; // 128B/cycle
    private const int MMA_PRIM_M = 32;
    private const int MMA_PRIM_N = 32;
    private const int MMA_PRIM_K = 32;
    private const int MMA_PRIM_CYCLES = 8;

    private readonly int _loopsCount;
    private readonly int _reductionLoopsCount;
    private readonly int _buffersCount;
    private readonly LoopMask[] _loopMasks;

    private readonly Solver _solver = new("TilingSolver");
    private readonly IntVar[] _tiles;
    private readonly IntVar[,] _orders;
    private readonly IntVar[,] _places;
    private readonly OrderCombination[][] _orderCombinations;

    private readonly IntVar _objective;
    private readonly DecisionBuilder _decisionBuilder;
    private readonly SolutionCollector _solutionCollector;

    private readonly IntExpr[] _dims;
    private readonly IntExpr[] _tileCounts;

    public TilingSolver(int[] dims, int buffersCount, LoopMask[] loopMasks)
    {
        // 1. Constants
        _loopsCount = dims.Length;
        _reductionLoopsCount = _loopsCount - loopMasks[^1].Count;
        _buffersCount = buffersCount;
        _loopMasks = loopMasks;

        // 2. Variables
        _tiles = CreateTileVars(dims);
        _orders = CreateLoopOrderVars();
        _places = CreateBufferPlaceVars();

        // 3. Expressions
        // 3.1. Orders
        _dims = dims.Select(x => _solver.MakeIntConst(x)).ToArray();
        _orderCombinations = CreateOrderCombinationExprs();
        _tileCounts = CreateTileCountsExprs();

        // 3.2. Buffer sizes
        var bufferSizes = CreateBufferSizeExprs();
        var totalBufferSize = bufferSizes.Aggregate((IntExpr)_solver.MakeIntConst(0), (x, y) => x + y);

        // 3.3. Memory access latency
        var bufferTileAccessCycles = bufferSizes.Select(x => x.CeilDiv(L3_BANDWIDTH)).ToArray();
        var bufferAccessTimes = CreateBufferAccessTimesExprs();
        var bufferAccessCycles = bufferTileAccessCycles.Zip(bufferAccessTimes).Select(x => x.First * x.Second).ToArray();
        var totalMemoryAccessCycles = bufferAccessCycles.Aggregate((IntExpr)_solver.MakeIntConst(0), (x, y) => x + y);

        // 3.4. Calc latency
        var tileCalcCycles = GetTileCalcCycles(_tiles);
        var totalCalcCyles = _tileCounts.Aggregate(tileCalcCycles, (x, y) => x * y);

        var totalCycles = _solver.MakeMax(totalMemoryAccessCycles, totalCalcCyles);

        // 4. Constraints
        // 4.1. Buffer size
        _solver.Add(totalBufferSize <= L2_SIZE);

        // 4.2. Orders
        AddOrdersConstraints();

        // 4.3. Places
        AddPlacesConstraints();

        // 4.4. Reduction aware
        AddReductionPlacesConstraints();

        // 5. Objective
        _objective = totalCycles.Var();

        var allVars = _tiles.Concat(_orders.Cast<IntVar>()).Concat(_places.Cast<IntVar>()).ToArray();
        _decisionBuilder = _solver.MakePhase(allVars, Solver.CHOOSE_FIRST_UNBOUND, Solver.ASSIGN_MIN_VALUE);
        _solutionCollector = _solver.MakeLastSolutionCollector();
        _solutionCollector.Add(allVars);
        _solutionCollector.AddObjective(_objective);
    }

    public void Solve()
    {
        var objeciveMonitor = _solver.MakeMinimize(_objective, 1);
        var searchLog = _solver.MakeSearchLog(100000, objeciveMonitor);
        var searchLimit = _solver.MakeImprovementLimit(_objective, false, 1, 0, 1, 2);
        var timeLimit = _solver.MakeTimeLimit(5000);

        _solver.Solve(_decisionBuilder, new SearchMonitor[] { objeciveMonitor, searchLimit, timeLimit, searchLog, _solutionCollector });

        if (_solutionCollector.SolutionCount() < 1)
        {
            throw new InvalidOperationException();
        }

        var solution = _solutionCollector.SolutionCount() - 1;
        var getLoopName = (int loop) => "mnk"[GetOrderDim(solution, loop)];

        Debug.WriteLine($"(tm, tn, tk): ({_solutionCollector.Value(solution, _tiles[0])}, {_solutionCollector.Value(solution, _tiles[1])}, {_solutionCollector.Value(solution, _tiles[2])})");
        Debug.WriteLine($"for ({getLoopName(0)}, {getLoopName(1)}, {getLoopName(2)})");
        Debug.WriteLine($"buffer place(a, b, c): ({GetBufferPlace(solution, 0)}, {GetBufferPlace(solution, 1)}, {GetBufferPlace(solution, 2)})");
    }

    private static IntExpr GetTileCalcCycles(IntVar[] tiles)
    {
        return tiles[0].CeilDiv(MMA_PRIM_M) * tiles[1].CeilDiv(MMA_PRIM_N) * tiles[2].CeilDiv(MMA_PRIM_K) * MMA_PRIM_CYCLES;
    }

    private IntVar[] CreateTileVars(int[] upperBounds)
    {
        var tiles = new IntVar[upperBounds.Length];
        for (int i = 0; i < tiles.Length; i++)
        {
            tiles[i] = _solver.MakeIntVar(1, upperBounds[i], $"t{i}");
        }

        return tiles;
    }

    private IntVar[,] CreateLoopOrderVars()
    {
        var orders = new IntVar[_loopsCount, _loopsCount];
        for (int i = 0; i < _loopsCount; i++)
        {
            for (int j = 0; j < _loopsCount; j++)
            {
                orders[i, j] = _solver.MakeBoolVar($"order_d{i}_l{j}");
            }
        }

        return orders;
    }

    private IntVar[,] CreateBufferPlaceVars()
    {
        var places = new IntVar[_buffersCount, _loopsCount + 1];
        for (int i = 0; i < _buffersCount; i++)
        {
            for (int j = 0; j < _loopsCount + 1; j++)
            {
                places[i, j] = _solver.MakeBoolVar($"place_b{i}_{j}");
            }
        }

        return places;
    }

    private OrderCombination[][] CreateOrderCombinationExprs()
    {
        var maxCount = _loopsCount + 1;
        var permutations = new OrderCombination[maxCount][];
        for (int i = 0; i < permutations.Length; i++)
        {
            permutations[i] = CreateOrderCombinationExprs(i);
        }

        return permutations;
    }

    private OrderCombination[] CreateOrderCombinationExprs(int count)
    {
        var combinations = new OrderCombination[MathUtility.Factorial(_loopsCount) / (MathUtility.Factorial(_loopsCount - count) * MathUtility.Factorial(count))];

        int index = 0;
        var combination = new int[count];
        bool[] chosen = new bool[_loopsCount];
        GenerateOrderCombinations(count, combinations, combination, 0, 0, chosen, ref index);
        return combinations;
    }

    private void GenerateOrderCombinations(int count, OrderCombination[] combinations, int[] combination, int start, int index, bool[] chosen, ref int combineResultIndex)
    {
        if (index == count)
        {
#if true
            Debug.WriteLine($"{count}: {string.Join(", ", combination)}");
#endif
            int[] permutation = new int[count];
            int permuteResultIndex = 0;
            ref var result = ref combinations[combineResultIndex++];
            result = new OrderCombination(CombinationToLoopMask(combination));
            GenerateOrderPermutations(ref result, combination, permutation, 0, chosen, ref permuteResultIndex);
            return;
        }

        for (int i = start; i <= _loopsCount - count + index; ++i)
        {
            combination[index] = i;
            GenerateOrderCombinations(count, combinations, combination, i + 1, index + 1, chosen, ref combineResultIndex);
        }
    }

    private LoopMask CombinationToLoopMask(int[] combination)
    {
        uint mask = 0;
        foreach (var loop in combination)
        {
            mask |= 1U << loop;
        }

        return new LoopMask(mask);
    }

    private void GenerateOrderPermutations(ref OrderCombination result, int[] combination, int[] permutation, int index, bool[] chosen, ref int permuteResultIndex)
    {
        if (index == combination.Length)
        {
#if true
            Debug.WriteLine($"{string.Join(", ", permutation)}");
#endif
            if (combination.Length == 0)
            {
                result.Expr = _solver.MakeIntConst(1);
            }
            else
            {
                IntExpr? expr = null;
                for (int i = 0; i < permutation.Length; i++)
                {
                    var order = _orders[permutation[i], i];
                    expr = expr == null ? order : expr * order;
                }

                result.Expr = result.Expr == null ? expr! : result.Expr + expr;
            }

            return;
        }

        for (int i = 0; i < combination.Length; ++i)
        {
            if (!chosen[i])
            {
                chosen[i] = true;
                permutation[index] = combination[i];
                GenerateOrderPermutations(ref result, combination, permutation, index + 1, chosen, ref permuteResultIndex);
                chosen[i] = false;
            }
        }
    }

    private int GetOrderDim(int solution, int order)
    {
        for (int i = 0; i < _loopsCount; i++)
        {
            if (_solutionCollector.Value(solution, _orders[i, order]) == 1)
            {
                return i;
            }
        }

        throw new InvalidOperationException();
    }

    private int GetBufferPlace(int solution, int bufferIndex)
    {
        for (int i = 0; i < _loopsCount + 1; i++)
        {
            if (_solutionCollector.Value(solution, _places[bufferIndex, i]) == 1)
            {
                return i;
            }
        }

        throw new InvalidOperationException();
    }

    private IntExpr[] CreateTileCountsExprs()
    {
        var exprs = new IntExpr[_loopsCount];
        for (int i = 0; i < exprs.Length; i++)
        {
            exprs[i] = _dims[i].CeilDiv(_tiles[i]);
        }

        return exprs;
    }

    private IntExpr[] CreateBufferSizeExprs()
    {
        var exprs = new IntExpr[_buffersCount];
        for (int i = 0; i < exprs.Length; i++)
        {
            exprs[i] = CreateBufferSizeExpr(i);
        }

        return exprs;
    }

    private IntExpr CreateBufferSizeExpr(int bufferIndex)
    {
        var loopMask = _loopMasks[bufferIndex];
        IntExpr? bufferSizeExpr = null;
        for (int place = 0; place < _loopsCount + 1; place++)
        {
            IntExpr? placedBufferSizeExpr = null;
            foreach (var combination in _orderCombinations[place])
            {
                IntExpr? tileSizeExpr = null;
                for (int loop = 0; loop < _loopsCount; loop++)
                {
                    if (loopMask.IsRelated(loop))
                    {
                        var tileDimExpr = combination.Loops.IsRelated(loop) ? _tiles[loop] : _dims[loop];
                        tileSizeExpr = tileSizeExpr == null ? tileDimExpr : tileSizeExpr * tileDimExpr;
                    }
                }

                var gatedTileSizeExpr = combination.Expr * tileSizeExpr;
                placedBufferSizeExpr = placedBufferSizeExpr == null ? gatedTileSizeExpr : placedBufferSizeExpr + gatedTileSizeExpr;
            }

            var gatedPlacedBufferSizeExpr = _places[bufferIndex, place] * placedBufferSizeExpr;
            bufferSizeExpr = bufferSizeExpr == null ? gatedPlacedBufferSizeExpr : bufferSizeExpr + gatedPlacedBufferSizeExpr;
        }

        return bufferSizeExpr * sizeof(float);
    }

    private IntExpr[] CreateBufferAccessTimesExprs()
    {
        var exprs = new IntExpr[_buffersCount];
        for (int i = 0; i < exprs.Length; i++)
        {
            exprs[i] = CreateBufferAccessTimesExpr(i);
        }

        return exprs;
    }

    private IntExpr CreateBufferAccessTimesExpr(int bufferIndex)
    {
        IntExpr? timesExpr = null;
        for (int place = 0; place < _loopsCount + 1; place++)
        {
            IntExpr? placedTimesExpr = null;
            foreach (var combination in _orderCombinations[place])
            {
                IntExpr timeExpr = _solver.MakeIntConst(1);
                for (int loop = 0; loop < _loopsCount; loop++)
                {
                    if (combination.Loops.IsRelated(loop))
                    {
                        timeExpr *= _tileCounts[loop];
                    }
                }

                var gatedTimesExpr = combination.Expr * timeExpr;
                placedTimesExpr = placedTimesExpr == null ? gatedTimesExpr : placedTimesExpr + gatedTimesExpr;
            }

            var gatedPlacedTimesExpr = _places[bufferIndex, place] * placedTimesExpr;
            timesExpr = timesExpr == null ? gatedPlacedTimesExpr : timesExpr + gatedPlacedTimesExpr;
        }

        return timesExpr!;
    }

    private void AddOrdersConstraints()
    {
        // 1. Every dim has one loop
        for (int i = 0; i < _loopsCount; i++)
        {
            IntExpr expr = _orders[i, 0];
            for (int j = 1; j < _loopsCount; j++)
            {
                expr += _orders[i, j];
            }

            _solver.Add(expr == 1);
        }

        // 2. Every loop has one dim
        for (int i = 0; i < _loopsCount; i++)
        {
            IntExpr expr = _orders[0, i];
            for (int j = 1; j < _loopsCount; j++)
            {
                expr += _orders[j, i];
            }

            _solver.Add(expr == 1);
        }
    }

    private void AddPlacesConstraints()
    {
        // 1. Every buffer has one place
        for (int i = 0; i < _buffersCount; i++)
        {
            IntExpr expr = _places[i, 0];
            for (int j = 1; j < _loopsCount + 1; j++)
            {
                expr += _places[i, j];
            }

            _solver.Add(expr == 1);
        }
    }

    private void AddReductionPlacesConstraints()
    {
        for (int place = 1; place < _loopsCount + 1; place++)
        {
            var placeVar = _places[_buffersCount - 1, place];

            // Outer loops should not be reduction loops.
            IntExpr? anyOrder = null;
            for (int reductionLoop = _loopsCount - _reductionLoopsCount; reductionLoop < _loopsCount; reductionLoop++)
            {
                for (int order = 0; order < place; order++)
                {
                    var expr = _orders[reductionLoop, order];
                    anyOrder = anyOrder == null ? expr : anyOrder + expr;
                }
            }

            var constraint = placeVar * (anyOrder ?? _solver.MakeIntConst(1)) == 0;
            _solver.Add(constraint);
        }
    }
}

public static class CPExtensions
{
    public static IntExpr CeilDiv(this IntExpr numer, long denom) =>
        (numer + (denom - 1)) / denom;

    public static IntExpr CeilDiv(this IntExpr numer, IntExpr denom) =>
        denom.solver().MakeDiv(numer + (denom - 1), denom);

    public static IntExpr CeilDiv(this long numer, IntExpr denom) =>
        denom.solver().MakeDiv(numer + (denom - 1), denom);

    public static IntExpr CeilDiv(this int numer, IntExpr denom) =>
        denom.solver().MakeDiv(numer + (denom - 1), denom);
}
