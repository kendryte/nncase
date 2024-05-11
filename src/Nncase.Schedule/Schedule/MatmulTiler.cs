// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System.Text;
using CommunityToolkit.HighPerformance;
using Google.OrTools.ConstraintSolver;

namespace Nncase.Schedule;

public static class MatmulTiler
{
    public static void Solve()
    {
        const int TotalLevel = 2;
        var dimValues = new int[3] { 2048, 384, 8192 };
        var domain = new char[3] { 'm', 'n', 'k' };
        var tensors = new VTensor[3] {
            new VTensor("A", new[] { 'm', 'k' }),
            new VTensor("B", new[] { 'k', 'n' }),
            new VTensor("C", new[] { 'm', 'n' }),
        };

        // note reverse domain!
        Array.Reverse(dimValues);
        Array.Reverse(domain);

        var model = new Solver("tiling");
        var tileVars = new IntVar[TotalLevel + 1, domain.Length];
        var allVars = new List<IntVar>();
        var memoryCapacitys = new int[] { 65536, 4194304, int.MaxValue };
        var memoryBandWidths = new int[] { 128 /* b/cycle */, 16 /* b/cycle */, 4 };

        // create tilesize vars.
        // the reads table save the buffer read times, we can use it directly.
        var readsTable = new IntExpr[tensors.Length, TotalLevel];
        IntExpr one = model.MakeIntConst(1, "one");
        for (int l = 0; l < TotalLevel; l++)
        {
            var mul = one;
            for (int i = 0; i < domain.Length; i++)
            {
                tileVars[l, i] = model.MakeIntVar(1, dimValues[i], $"T_{domain[i]}{l + 1}");
                allVars.Add(tileVars[l, i]);
                mul *= tileVars[l, i];
            }

            for (int i = 0; i < tensors.Length; i++)
            {
                readsTable[i, l] = mul;
                if (l > 0)
                {
                    readsTable[i, l] *= readsTable[i, l - 1];
                }
            }
        }

        // we save the statement level tile size var in the last row.
        for (int i = 0; i < domain.Length; i++)
        {
            tileVars[TotalLevel, i] = model.MakeIntVar(1, dimValues[i], $"T_{domain[i]}");
            allVars.Add(tileVars[TotalLevel, i]);

            for (int a = 0; a < tensors.Length; a++)
            {
                for (int l = 0; l < TotalLevel; l++)
                {
                    if (tensors[a].Dims.Contains(domain[i]))
                    {
                        readsTable[a, l] *= tileVars[TotalLevel, i];
                    }
                }
            }
        }

        IntExpr elem = model.MakeIntConst(sizeof(float), "elem");
        var primitiveBufferSizes = new IntExpr[tensors.Length];
        for (int a = 0; a < tensors.Length; a++)
        {
            primitiveBufferSizes[a] = elem;
            for (int i = 0; i < domain.Length; i++)
            {
                if (tensors[a].Dims.Contains(domain[i]))
                {
                    primitiveBufferSizes[a] *= tileVars[TotalLevel, i];
                }
            }
        }

        // create buffer placement: gates(l,i,sl) mean create buffer size by memory level l, loop i, then store it into memory level sl.
        var placeGates = new IntVar[tensors.Length, TotalLevel, domain.Length][];
        var dataReads = new IntExpr[tensors.Length, TotalLevel, domain.Length][];
        var dataWrites = new IntExpr[tensors.Length, TotalLevel, domain.Length][];
        var bufferSizes = new IntExpr[tensors.Length, TotalLevel, domain.Length];
        for (int ts = 0; ts < tensors.Length; ts++)
        {
            for (int l = 0; l < TotalLevel; l++)
            {
                for (int i = 0; i < domain.Length; i++)
                {
                    {
                        var lastLevelSize = (l == 0 && i == 0) ? primitiveBufferSizes[ts] : (i == 0 ? bufferSizes[ts, l - 1, domain.Length - 1] : bufferSizes[ts, l, i - 1]);
                        bufferSizes[ts, l, i] = tensors[ts].Dims.Contains(domain[i]) ? lastLevelSize * tileVars[l, i] : lastLevelSize;
                    }

                    // we can create buffer size by high loop level, but put it into low loop level.
                    var subLevelPlace = placeGates[ts, l, i] = new IntVar[l + 1];
                    var subLevelReads = dataReads[ts, l, i] = new IntExpr[l + 1];
                    var subLevelWrites = dataWrites[ts, l, i] = new IntExpr[l + 1];
                    for (int sl = 0; sl < l + 1; sl++)
                    {
                        subLevelPlace[sl] = model.MakeBoolVar($"place({tensors[ts].Name}, {l + 1}, {domain[i]}, {sl + 1})");
                        allVars.Add(subLevelPlace[sl]);

                        // 1. compute data reads
                        if (sl == 0)
                        {
                            // get reads from table when store buffer at lowest level.
                            subLevelReads[sl] = readsTable[ts, l];
                        }
                        else
                        {
                            // currently reads need multiply the lower level write times
                            subLevelReads[sl] = one;
                            for (int ci = 0; ci < domain.Length; ci++)
                            {
                                subLevelReads[sl] *= tileVars[l, ci];
                            }

                            for (int ci = 0; ci < domain.Length; ci++)
                            {
                                subLevelReads[sl] *= placeGates[ts, l - 1, ci][l - 1] * dataWrites[ts, l - 1, ci][l - 1];
                            }
                        }

                        // 2. compute data writes
                        subLevelWrites[sl] = bufferSizes[ts, l, i];
                        for (int ni = i + 1; ni < domain.Length; ni++)
                        {
                            subLevelWrites[sl] *= tileVars[l, ni];
                        }
                    }
                }
            }
        }

        // add placement constraints.
        // 1. must store one buffer at lowest level.
        var lowestLevelBufferConstraints = new Constraint[tensors.Length];
        for (int a = 0; a < tensors.Length; a++)
        {
            IntExpr lowestBufferNums = model.MakeIntConst(0, "zero");
            for (int l = 0; l < TotalLevel; l++)
            {
                for (int i = 0; i < domain.Length; i++)
                {
                    lowestBufferNums += placeGates[a, l, i][0];
                }
            }

            lowestLevelBufferConstraints[a] = model.MakeEquality(lowestBufferNums, 1);
            model.Add(lowestLevelBufferConstraints[a]);
        }

        // 2. each tensor only can create one or zero buffer at each create level.
        var eachlevelCreateBufferConstraints = new Constraint[tensors.Length, TotalLevel];
        var eachlevelCreateBufferNums = new IntExpr[tensors.Length, TotalLevel];
        for (int a = 0; a < tensors.Length; a++)
        {
            for (int l = 0; l < TotalLevel; l++)
            {
                IntExpr bufferNums = model.MakeIntConst(0, "zero");
                for (int i = 0; i < domain.Length; i++)
                {
                    var slGates = placeGates[a, l, i];
                    for (int sl = 0; sl < slGates.Length; sl++)
                    {
                        bufferNums += slGates[sl];
                    }
                }

                eachlevelCreateBufferNums[a, l] = bufferNums;
                eachlevelCreateBufferConstraints[a, l] = model.MakeLessOrEqual(bufferNums, one);
                model.Add(eachlevelCreateBufferConstraints[a, l]);
            }
        }

        // 3. if current level has create a buffer, it's requires previous level store a buffer.
        var depLevelBufferConstraints = new Constraint[tensors.Length, TotalLevel - 1];
        for (int a = 0; a < tensors.Length; a++)
        {
            for (int l = 0; l < TotalLevel - 1; l++)
            {
                IntExpr previousLevelStoreBufferNums = model.MakeIntConst(0, "zero");
                for (int pl = l + 1; pl < TotalLevel; pl++)
                {
                    for (int i = 0; i < domain.Length; i++)
                    {
                        previousLevelStoreBufferNums += placeGates[a, pl, i][l];
                    }
                }

                depLevelBufferConstraints[a, l] = model.MakeGreaterOrEqual(previousLevelStoreBufferNums, model.MakeIsEqualVar(eachlevelCreateBufferNums[a, l], one));
                model.Add(depLevelBufferConstraints[a, l]);
            }
        }

        // add tile vars equal to domain value constraints
        var tileVarConstraints = new Constraint[domain.Length];
        for (int i = 0; i < domain.Length; i++)
        {
            var sp = tileVars.AsSpan2D();
            var column = sp.GetColumn(i).ToArray();
            IntExpr prod = column[0];
            for (int l = 1; l < column.Length; l++)
            {
                prod *= column[l];
            }

            tileVarConstraints[i] = model.MakeEquality(prod, dimValues[i]);
            model.Add(tileVarConstraints[i]);
        }

        // add the memory capacity constraints
        var levelBufferSizes = Enumerable.Range(0, TotalLevel).Select(i => (IntExpr)model.MakeIntConst(0, "zero")).ToArray();
        for (int a = 0; a < tensors.Length; a++)
        {
            for (int l = 0; l < TotalLevel; l++)
            {
                for (int i = 0; i < domain.Length; i++)
                {
                    // we can create buffer size by high loop level, but put it into low loop level.
                    for (int sl = 0; sl < l + 1; sl++)
                    {
                        levelBufferSizes[sl] += placeGates[a, l, i][sl] * bufferSizes[a, sl, i];
                    }
                }
            }
        }

        var capacityConstraints = new Constraint[TotalLevel];
        for (int l = 0; l < TotalLevel; l++)
        {
            capacityConstraints[l] = model.MakeLessOrEqual(levelBufferSizes[l], memoryCapacitys[l]);
            model.Add(capacityConstraints[l]);
        }

        // compute the cycles as objective
        var levelCycles = Enumerable.Range(0, TotalLevel).Select(i => (IntExpr)model.MakeIntConst(0, "zero")).ToArray();
        for (int a = 0; a < tensors.Length; a++)
        {
            for (int l = 0; l < TotalLevel; l++)
            {
                for (int i = 0; i < domain.Length; i++)
                {
                    // we can create buffer size by high loop level, but put it into low loop level.
                    for (int sl = 0; sl < l + 1; sl++)
                    {
                        levelCycles[sl] += placeGates[a, l, i][sl] * (dataWrites[a, l, i][sl] + dataReads[a, l, i][sl]);
                    }
                }
            }
        }

        for (int l = 0; l < TotalLevel; l++)
        {
            levelCycles[l] = model.MakeDiv(levelCycles[l] + (memoryBandWidths[l] - 1), memoryBandWidths[l]);
        }

        var totalCycles = levelCycles[0];
        for (int l = 1; l < TotalLevel; l++)
        {
            totalCycles = model.MakeMax(totalCycles, levelCycles[l]);
        }

        var totalCyclesVar = totalCycles.Var();
        totalCyclesVar.SetRange(1, long.MaxValue / memoryBandWidths[0]); /* avoid crash. */

        var objectiveMonitor = model.MakeMinimize(totalCyclesVar, 1);
        var logger = model.MakeSearchTrace("tiling:");
        var collector = model.MakeLastSolutionCollector();
        collector.Add(allVars.ToArray());
        collector.Add(levelBufferSizes.Select(c => c.Var()).ToArray());
        collector.Add(capacityConstraints.Select(c => c.Var()).ToArray());
        for (int ts = 0; ts < tensors.Length; ts++)
        {
            for (int l = 0; l < TotalLevel; l++)
            {
                for (int i = 0; i < domain.Length; i++)
                {
                    collector.Add(bufferSizes[ts, l, i].Var());
                    for (int sl = 0; sl < l - 1; sl++)
                    {
                        collector.Add(dataReads[ts, l, i][sl].Var());
                        collector.Add(dataWrites[ts, l, i][sl].Var());
                    }
                }
            }
        }

        collector.AddObjective(totalCyclesVar);
        var decisionBuilder = model.MakePhase(allVars.ToArray(), Solver.CHOOSE_LOWEST_MIN, Solver.ASSIGN_MIN_VALUE);
        var status = model.Solve(decisionBuilder, new SearchMonitor[] { collector, objectiveMonitor });
        System.Console.WriteLine($"solve status: {status}");

        if (status)
        {
            var sol = collector.Solution(0);
            StreamWriter writer = new(Diagnostics.DumpScope.Current.OpenFile("tiled.py"));
            void DisplayAlloc(string indent, int level, int loop)
            {
                for (int a = 0; a < tensors.Length; a++)
                {
                    var storeGates = placeGates[a, level, loop];
                    for (int sl = 0; sl < storeGates.Length; sl++)
                    {
                        if (sol.Value(storeGates[sl]) == 1)
                        {
                            var tensor = tensors[a];
                            var sizeSS = new List<string>();
                            for (int d = 0; d < tensor.Dims.Length; d++)
                            {
                                var ss = new List<string>();
                                for (int l = 0; l <= sl; l++)
                                {
                                    for (int i = 0; i < ((l == sl) ? loop + 1 : domain.Length); i++)
                                    {
                                        if (tileVars[l, i].Name().Contains(tensor.Dims[d], StringComparison.CurrentCulture))
                                        {
                                            ss.Add(tileVars[l, i].Name());
                                        }
                                    }
                                }

                                for (int i = 0; i < domain.Length; i++)
                                {
                                    if (tileVars[TotalLevel, i].Name().Contains(tensor.Dims[d], StringComparison.CurrentCulture))
                                    {
                                        ss.Add(tileVars[TotalLevel, i].Name());
                                    }
                                }

                                sizeSS.Add(string.Join("*", ss));
                            }

                            writer.WriteLine($"{indent}sub_{tensor.Name} = {tensor.Name}[{string.Join(",", sizeSS)}] @ L{sl + 1}");
                        }
                    }
                }
            }

            string indent = string.Empty;
            for (int l = TotalLevel - 1; l >= 0; l--)
            {
                writer.WriteLine($"{indent}L{l + 1}:");
                for (int i = domain.Length - 1; i >= 0; i--)
                {
                    DisplayAlloc(indent, l, i);
                    writer.WriteLine($"{indent}for {domain[i]}{l + 1} in range(T_{domain[i]}{l + 1} = {sol.Value(tileVars[l, i])}):");
                    indent += "  ";
                }
            }

            // k,n,m
            writer.WriteLine($"{indent} C[{sol.Value(tileVars[TotalLevel, 2])},{sol.Value(tileVars[TotalLevel, 1])}] += A[{sol.Value(tileVars[TotalLevel, 2])},{sol.Value(tileVars[TotalLevel, 0])}] * B[{sol.Value(tileVars[TotalLevel, 0])},{sol.Value(tileVars[TotalLevel, 1])}]");

            writer.Close();

            // check constrains
            for (int l = 0; l < TotalLevel; l++)
            {
                System.Console.WriteLine($"L{l + 1} used : {sol.Value(capacityConstraints[0])}");
            }
        }
    }
}

internal sealed record VTensor(string Name, char[] Dims)
{
    public bool ContainsLoop(string loopName)
    {
        foreach (var dim in Dims)
        {
            if (loopName.Contains(dim, System.StringComparison.Ordinal))
            {
                return true;
            }
        }

        return false;
    }
}
