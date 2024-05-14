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
        var primtives = new int[3] { 32, 32, 32 };
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
        IntExpr one = model.MakeIntConst(1, "one");
        IntExpr zero = model.MakeIntConst(0, "zero");
        for (int l = 0; l < TotalLevel; l++)
        {
            for (int i = 0; i < domain.Length; i++)
            {
                tileVars[l, i] = model.MakeIntVar(1, dimValues[i] / primtives[i], $"T_{domain[i]}{l + 1}");
                allVars.Add(tileVars[l, i]);
            }
        }

        // we save the statement level tile size var in the last row.
        for (int i = 0; i < domain.Length; i++)
        {
            tileVars[TotalLevel, i] = model.MakeIntVar(1, dimValues[i] / primtives[i], $"T_{domain[i]}");
            allVars.Add(tileVars[TotalLevel, i]);
        }

        // and the reads table save the buffer read times, we can use it directly.
        IntExpr elem = model.MakeIntConst(sizeof(float), "elem");
        var primitiveBufferSizes = new IntExpr[tensors.Length];
        for (int a = 0; a < tensors.Length; a++)
        {
            primitiveBufferSizes[a] = elem;
            for (int i = 0; i < domain.Length; i++)
            {
                if (tensors[a].Dims.Contains(domain[i]))
                {
                    primitiveBufferSizes[a] *= tileVars[TotalLevel, i] * primtives[i];
                }
            }
        }

        IntExpr totalLoopTimes = Enumerable.Range(0, TotalLevel).Select(i => Enumerable.Range(0, domain.Length).Select(j => tileVars[i, j])).SelectMany(i => i).Aggregate(one, (acc, tileVar) => acc * tileVar);
        var readsTable = Enumerable.Range(0, tensors.Length).Select(a => primitiveBufferSizes[a] * totalLoopTimes).ToArray();

        // create buffer placement: gates(l,i,sl) mean create buffer size by memory level l, loop i, then store it into memory level sl.
        var placeGates = new IntVar[tensors.Length, TotalLevel, domain.Length][];
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
                    var subLevelWrites = dataWrites[ts, l, i] = new IntExpr[l + 1];
                    for (int sl = 0; sl < l + 1; sl++)
                    {
                        subLevelPlace[sl] = model.MakeBoolVar($"place({tensors[ts].Name}, {l + 1}, {domain[i]}, {sl + 1})");
                        allVars.Add(subLevelPlace[sl]);

                        // 2. compute data writes
                        subLevelWrites[sl] = bufferSizes[ts, l, i];
                        for (int nl = l; nl < TotalLevel; nl++)
                        {
                            for (int ci = 0; ci < domain.Length; ci++)
                            {
                                if (nl == l && ci <= i)
                                {
                                    continue;
                                }

                                subLevelWrites[sl] *= tileVars[nl, ci];
                            }
                        }
                    }
                }
            }
        }

        // in each create level create data reads.
        var dataReads = new IntExpr[tensors.Length, TotalLevel];
        for (int a = 0; a < tensors.Length; a++)
        {
            for (int l = 0; l < TotalLevel; l++)
            {
                if (l == 0)
                {
                    // l0 reads should follow the computation.
                    dataReads[a, l] = readsTable[a];
                }
                else
                {
                    dataReads[a, l] = zero;
                    for (int i = 0; i < domain.Length; i++)
                    {
                        var lowerLevelWrites = dataWrites[a, l - 1, i];
                        var lowerLevelGates = placeGates[a, l - 1, i];
                        for (int sl = 0; sl < lowerLevelWrites.Length; sl++)
                        {
                            dataReads[a, l] += lowerLevelGates[sl] * lowerLevelWrites[sl];
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
            IntExpr lowestBufferNums = zero;
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
                IntExpr bufferNums = zero;
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
                IntExpr previousLevelStoreBufferNums = zero;
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

            tileVarConstraints[i] = model.MakeEquality(prod, dimValues[i] / primtives[i]);
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
                        levelBufferSizes[sl] += placeGates[a, l, i][sl] * bufferSizes[a, l, i];
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
        var levelCycles = Enumerable.Range(0, TotalLevel).Select(i => zero).ToArray();
        var levelDataReads = Enumerable.Range(0, TotalLevel).Select(i => zero).ToArray();
        var levelDataWrites = Enumerable.Range(0, TotalLevel).Select(i => zero).ToArray();
        for (int a = 0; a < tensors.Length; a++)
        {
            for (int l = 0; l < TotalLevel; l++)
            {
                levelDataReads[l] += dataReads[a, l];
                for (int i = 0; i < domain.Length; i++)
                {
                    // we can create buffer size by high loop level, but put it into low loop level.
                    for (int sl = 0; sl < l + 1; sl++)
                    {
                        levelDataWrites[sl] += placeGates[a, l, i][sl] * dataWrites[a, l, i][sl];
                        levelCycles[sl] += (placeGates[a, l, i][sl] * dataWrites[a, l, i][sl]) + dataReads[a, l];
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
        var collector = model.MakeNBestValueSolutionCollector(5, false);
        collector.Add(allVars.ToArray());
        collector.Add(levelBufferSizes.Select(c => c.Var()).ToArray());
        collector.Add(levelDataReads.Select(c => c.Var()).ToArray());
        collector.Add(levelDataWrites.Select(c => c.Var()).ToArray());
        for (int ts = 0; ts < tensors.Length; ts++)
        {
            for (int l = 0; l < TotalLevel; l++)
            {
                collector.Add(dataReads[ts, l].Var());
                for (int i = 0; i < domain.Length; i++)
                {
                    collector.Add(bufferSizes[ts, l, i].Var());
                    for (int sl = 0; sl < l - 1; sl++)
                    {
                        collector.Add(dataWrites[ts, l, i][sl].Var());
                    }
                }
            }
        }

        collector.AddObjective(totalCyclesVar);
        var decisionBuilder = model.MakePhase(allVars.ToArray(), Solver.INT_VAR_DEFAULT, Solver.INT_VALUE_DEFAULT);
        var status = model.Solve(decisionBuilder, new SearchMonitor[] { collector, objectiveMonitor });
        System.Console.WriteLine($"solve status: {status}");

        if (status)
        {
            var sol = collector.Solution(collector.SolutionCount() - 1);
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
                                for (int l = 0; l <= level; l++)
                                {
                                    for (int i = 0; i < ((l == level) ? loop + 1 : domain.Length); i++)
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

                            writer.Write($"{indent}sub_{tensor.Name} = {tensor.Name}[{string.Join(",", sizeSS)}] @ L{sl + 1}");
                            writer.WriteLine($" # size: {sol.Value(bufferSizes[a, level, loop].Var())}");
                        }
                    }
                }
            }

            string indent = string.Empty;
            for (int l = TotalLevel - 1; l >= 0; l--)
            {
                writer.WriteLine($"{indent}L{l + 1}: ");
                writer.WriteLine($"{indent}# stored buffer size: {sol.Value(levelBufferSizes[l].Var())}");
                writer.WriteLine($"{indent}# data reads: {sol.Value(levelDataReads[l].Var())}");
                writer.WriteLine($"{indent}# data writes: {sol.Value(levelDataWrites[l].Var())}");
                for (int i = domain.Length - 1; i >= 0; i--)
                {
                    DisplayAlloc(indent, l, i);
                    writer.WriteLine($"{indent}for {domain[i]}{l + 1} in range(T_{domain[i]}{l + 1} = {sol.Value(tileVars[l, i])}):");
                    indent += "  ";
                }
            }

            // k,n,m
            writer.WriteLine($"{indent} C[{sol.Value(tileVars[TotalLevel, 2]) * primtives[2]},{sol.Value(tileVars[TotalLevel, 1]) * primtives[1]}] += A[{sol.Value(tileVars[TotalLevel, 2]) * primtives[2]},{sol.Value(tileVars[TotalLevel, 0]) * primtives[0]}] * B[{sol.Value(tileVars[TotalLevel, 0]) * primtives[0]},{sol.Value(tileVars[TotalLevel, 1]) * primtives[1]}]");

            writer.Close();
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
