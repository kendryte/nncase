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

        var index = Enumerable.Range(0, domain.Length).ToArray();
        var perms = new[] { index, index, index }.CartesianProduct().Where(arr => new HashSet<int>(arr).Count == domain.Length).Select(arr => arr.ToArray()).ToArray();
        for (int i = 0; i < perms.Length; i++)
        {
            for (int j = 0; j < perms.Length; j++)
            {
                var newDomain = new char[TotalLevel + 1, domain.Length];
                for (int k = 0; k < domain.Length; k++)
                {
                    newDomain[0, k] = domain[perms[i][k]];
                    newDomain[1, k] = domain[perms[j][k]];
                    newDomain[2, k] = domain[k];
                }

                var count = (i * perms.Length) + j;
                if (count != 21)
                {
                    continue;
                }

                SolveWithOrder(TotalLevel, dimValues, newDomain, primtives, tensors, $"{count}");
            }
        }
    }

    private static void SolveWithOrder(int totalLevel, int[] dimValues, char[,] domain, int[] primtives, VTensor[] tensors, string prefix)
    {
        var model = new Solver("tiling");
        var tileVars = new IntVar[totalLevel + 1, domain.GetLength(1)];
        var allVars = new List<IntVar>();

        // var memoryCapacitys = new int[] { 65536, 4194304, int.MaxValue };
        var memoryCapacitys = new int[] { 2 * 1024 * 1024, int.MaxValue, int.MaxValue };
        var memoryBandWidths = new int[] { 1024 /* b/cycle */, 256 /* b/cycle */, 64 /* 204GB/s / 3.49GHz/s */ };

        // create tilesize vars.
        IntExpr one = model.MakeIntConst(1, "one");
        IntExpr zero = model.MakeIntConst(0, "zero");
        for (int l = 0; l < totalLevel; l++)
        {
            for (int i = 0; i < domain.GetLength(1); i++)
            {
                var j = domain.GetRowSpan(totalLevel).IndexOf(domain[l, i]);
                tileVars[l, i] = model.MakeIntVar(1, dimValues[j] / primtives[j], $"T_{domain[l, i]}{l + 1}");
                allVars.Add(tileVars[l, i]);
            }
        }

        // we save the statement level tile size var in the last row.
        for (int i = 0; i < domain.GetLength(1); i++)
        {
            tileVars[totalLevel, i] = model.MakeIntVar(1, dimValues[i] / primtives[i], $"T_{domain[totalLevel, i]}");
            allVars.Add(tileVars[totalLevel, i]);
        }

        // and the reads table save the buffer read times, we can use it directly.
        IntExpr elem = model.MakeIntConst(sizeof(float), "elem");
        var primitiveBufferSizes = new IntExpr[tensors.Length];
        for (int a = 0; a < tensors.Length; a++)
        {
            primitiveBufferSizes[a] = elem;
            for (int i = 0; i < domain.GetLength(1); i++)
            {
                if (tensors[a].Dims.Contains(domain[totalLevel, i]))
                {
                    primitiveBufferSizes[a] *= tileVars[totalLevel, i] * primtives[i];
                }
            }
        }

        IntExpr totalLoopTimes = Enumerable.Range(0, totalLevel).Select(i => Enumerable.Range(0, domain.GetLength(1)).Select(j => tileVars[i, j])).SelectMany(i => i).Aggregate(one, (acc, tileVar) => acc * tileVar);
        var readsTable = Enumerable.Range(0, tensors.Length).Select(a => primitiveBufferSizes[a] * totalLoopTimes).ToArray();

        // create buffer placement: gates(l,i,sl) mean create buffer size by memory level l, loop i, then store it into memory level sl.
        var placeGates = new IntVar[tensors.Length, totalLevel, domain.GetLength(1)][];
        var dataWrites = new IntExpr[tensors.Length, totalLevel, domain.GetLength(1)][];
        var bufferSizes = new IntExpr[tensors.Length, totalLevel, domain.GetLength(1)];
        for (int ts = 0; ts < tensors.Length; ts++)
        {
            for (int l = 0; l < totalLevel; l++)
            {
                for (int i = 0; i < domain.GetLength(1); i++)
                {
                    {
                        var lastLevelSize = (l == 0 && i == 0) ? primitiveBufferSizes[ts] : (i == 0 ? bufferSizes[ts, l - 1, domain.GetLength(1) - 1] : bufferSizes[ts, l, i - 1]);
                        bufferSizes[ts, l, i] = tensors[ts].Dims.Contains(domain[l, i]) ? lastLevelSize * tileVars[l, i] : lastLevelSize;
                    }

                    // we can create buffer size by high loop level, but put it into low loop level.
                    var subLevelPlace = placeGates[ts, l, i] = new IntVar[l + 1];
                    var subLevelWrites = dataWrites[ts, l, i] = new IntExpr[l + 1];
                    for (int sl = 0; sl < l + 1; sl++)
                    {
                        subLevelPlace[sl] = model.MakeBoolVar($"place({tensors[ts].Name}, {l}, {domain[l, i]}, {sl})");
                        allVars.Add(subLevelPlace[sl]);

                        // 2. compute data writes
                        subLevelWrites[sl] = bufferSizes[ts, l, i];
                        for (int nl = l; nl < totalLevel; nl++)
                        {
                            for (int ci = 0; ci < domain.GetLength(1); ci++)
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
        var dataReads = new IntExpr[tensors.Length, totalLevel];
        for (int a = 0; a < tensors.Length; a++)
        {
            for (int l = 0; l < totalLevel; l++)
            {
                if (l == 0)
                {
                    // l0 reads should follow the computation.
                    dataReads[a, l] = readsTable[a];
                }
                else
                {
                    dataReads[a, l] = zero;
                    for (int i = 0; i < domain.GetLength(1); i++)
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
        var lowestStoredBufferNums = new IntExpr[tensors.Length];
        for (int a = 0; a < tensors.Length; a++)
        {
            IntExpr lowestBufferNums = zero;
            for (int l = 0; l < totalLevel; l++)
            {
                for (int i = 0; i < domain.GetLength(1); i++)
                {
                    lowestBufferNums += placeGates[a, l, i][0];
                }
            }

            lowestStoredBufferNums[a] = lowestBufferNums;
            var c = model.MakeEquality(lowestStoredBufferNums[a], 1);
            c.SetName($"lowestStoredBufferNums[{tensors[a].Name}]");
            model.Add(c);
        }

        // 2. each tensor only can create one or zero buffer at each create level.
        var eachlevelCreateBufferConstraints = new Constraint[tensors.Length, totalLevel];
        var eachlevelCreateBufferNums = new IntExpr[tensors.Length, totalLevel];
        for (int a = 0; a < tensors.Length; a++)
        {
            for (int l = 0; l < totalLevel; l++)
            {
                IntExpr bufferNums = zero;
                for (int i = 0; i < domain.GetLength(1); i++)
                {
                    var slGates = placeGates[a, l, i];
                    for (int sl = 0; sl < slGates.Length; sl++)
                    {
                        bufferNums += slGates[sl];
                    }
                }

                eachlevelCreateBufferNums[a, l] = bufferNums;
                eachlevelCreateBufferConstraints[a, l] = model.MakeLessOrEqual(bufferNums, one);
                eachlevelCreateBufferConstraints[a, l].SetName($"eachlevelCreateBufferConstraints[{tensors[a].Name}, {l}]");
                model.Add(eachlevelCreateBufferConstraints[a, l]);
            }
        }

        // 3. if current level has create a buffer, it's requires previous level store a buffer.
        var depLevelBufferConstraints = new Constraint[tensors.Length, totalLevel - 1];
        for (int a = 0; a < tensors.Length; a++)
        {
            for (int l = 0; l < totalLevel - 1; l++)
            {
                IntExpr previousLevelStoreBufferNums = zero;
                for (int pl = l + 1; pl < totalLevel; pl++)
                {
                    for (int i = 0; i < domain.GetLength(1); i++)
                    {
                        previousLevelStoreBufferNums += placeGates[a, pl, i][pl];
                    }
                }

                depLevelBufferConstraints[a, l] = model.MakeGreaterOrEqual(previousLevelStoreBufferNums, model.MakeIsEqualVar(eachlevelCreateBufferNums[a, l], one));
                depLevelBufferConstraints[a, l].SetName($"depLevelBufferConstraints[{tensors[a].Name}, {l}]");
                model.Add(depLevelBufferConstraints[a, l]);
            }
        }

        // add tile vars equal to domain value constraints
        var tileVarConstraints = new Constraint[domain.GetLength(1)];
        for (int i = 0; i < domain.GetLength(1); i++)
        {
            IntExpr prod = tileVars[totalLevel, i];
            for (int l = 0; l < totalLevel; l++)
            {
                for (int j = 0; j < domain.GetLength(1); j++)
                {
                    if (domain[l, j] == domain[totalLevel, i])
                    {
                        prod *= tileVars[l, j];
                    }
                }
            }

            tileVarConstraints[i] = model.MakeEquality(prod, dimValues[i] / primtives[i]);
            model.Add(tileVarConstraints[i]);
        }

        // add the memory capacity constraints
        var levelBufferSizes = Enumerable.Range(0, totalLevel).Select(i => (IntExpr)model.MakeIntConst(0, "zero")).ToArray();
        for (int a = 0; a < tensors.Length; a++)
        {
            for (int l = 0; l < totalLevel; l++)
            {
                for (int i = 0; i < domain.GetLength(1); i++)
                {
                    // we can create buffer size by high loop level, but put it into low loop level.
                    for (int sl = 0; sl < l + 1; sl++)
                    {
                        levelBufferSizes[sl] += placeGates[a, l, i][sl] * bufferSizes[a, l, i];
                    }
                }
            }
        }

        var capacityConstraints = new Constraint[totalLevel];
        for (int l = 0; l < totalLevel; l++)
        {
            capacityConstraints[l] = model.MakeLessOrEqual(levelBufferSizes[l], memoryCapacitys[l]);
            model.Add(capacityConstraints[l]);
        }

        // compute the cycles as objective
        var levelCycles = Enumerable.Range(0, totalLevel).Select(i => zero).ToArray();
        var levelDataReads = Enumerable.Range(0, totalLevel).Select(i => zero).ToArray();
        var levelDataWrites = Enumerable.Range(0, totalLevel).Select(i => zero).ToArray();
        for (int a = 0; a < tensors.Length; a++)
        {
            for (int l = 0; l < totalLevel; l++)
            {
                levelDataReads[l] += dataReads[a, l];
                for (int i = 0; i < domain.GetLength(1); i++)
                {
                    // we can create buffer size by high loop level, but put it into low loop level.
                    for (int sl = 0; sl < l + 1; sl++)
                    {
                        // note we assume the top cache level is DDR, it's no need to count the data writes.
                        IntExpr writes;
                        if (sl != totalLevel - 1)
                        {
                            writes = placeGates[a, l, i][sl] * dataWrites[a, l, i][sl];
                        }
                        else
                        {
                            writes = zero;
                        }

                        levelDataWrites[sl] += writes;
                        levelCycles[sl] += writes + dataReads[a, l];
                    }
                }
            }
        }

        // divide the bandwidth.
        for (int l = 0; l < totalLevel; l++)
        {
            levelCycles[l] = model.MakeDiv(levelCycles[l] + (memoryBandWidths[l] - 1), memoryBandWidths[l]);
        }

        // custom the computation level cycles
        {
            // 1. load a and b, 476 GB/s
            var computationLoadAB = totalLoopTimes * (primitiveBufferSizes[0] + primitiveBufferSizes[1]);
            levelCycles[0] += model.MakeDiv(computationLoadAB + (memoryBandWidths[0] - 1), memoryBandWidths[0]);

            // 2. load c and store c, 10 GB/s
            var computationLoadStoreC = totalLoopTimes * (primitiveBufferSizes[2] + primitiveBufferSizes[2]);
            levelCycles[0] += model.MakeDiv(computationLoadStoreC + (memoryBandWidths[2] - 1), memoryBandWidths[2]);
        }

        var totalCycles = levelCycles[0];
        for (int l = 1; l < totalLevel; l++)
        {
            totalCycles = model.MakeMax(totalCycles, levelCycles[l]);
        }

        var totalCyclesVar = totalCycles.Var();
        totalCyclesVar.SetRange(1, long.MaxValue / memoryBandWidths[0]); /* avoid crash. */

        var objectiveMonitor = model.MakeMinimize(totalCyclesVar, 1);
        var logger = model.MakeSearchTrace($"{prefix}_tiling:");
        var collector = model.MakeNBestValueSolutionCollector(5, false);
        collector.Add(allVars.ToArray());
        collector.Add(lowestStoredBufferNums.Select(c => c.Var()).ToArray());
        collector.Add(eachlevelCreateBufferNums.AsSpan().ToArray().Select(c => c.Var()).ToArray());
        collector.Add(levelBufferSizes.Select(c => c.Var()).ToArray());
        collector.Add(levelDataReads.Select(c => c.Var()).ToArray());
        collector.Add(levelDataWrites.Select(c => c.Var()).ToArray());
        for (int ts = 0; ts < tensors.Length; ts++)
        {
            for (int l = 0; l < totalLevel; l++)
            {
                collector.Add(dataReads[ts, l].Var());
                for (int i = 0; i < domain.GetLength(1); i++)
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
        var decisionBuilder = model.MakeDefaultPhase(allVars.ToArray());
        var status = model.Solve(decisionBuilder, new SearchMonitor[] { collector, objectiveMonitor });
        System.Console.WriteLine($"solve status: {status}");

        if (status)
        {
            var sol = collector.Solution(collector.SolutionCount() - 1);
            StreamWriter writer = new(Stream.Null);
#if DEBUG
            writer = new(Diagnostics.DumpScope.Current.OpenFile($"{prefix}_tiled.py"));
#endif
            var tensorResides = new Dictionary<string, List<(string, int, int)>>();
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
                            var dims = new List<string>();
                            var offsets = new List<string>();
                            var (lastName, lastLevel, lastLoop) = tensorResides[tensor.Name].Last();

                            // compute sub tensor slice params.
                            for (int d = 0; d < tensor.Dims.Length; d++)
                            {
                                var varNames = new List<string>() { $"P_{tensor.Dims[d]}" };
                                for (int l = 0; l <= level; l++)
                                {
                                    for (int i = 0; i < ((l == level) ? loop + 1 : domain.GetLength(1)); i++)
                                    {
                                        if (tileVars[l, i].Name().Contains(tensor.Dims[d], StringComparison.CurrentCulture))
                                        {
                                            varNames.Add(tileVars[l, i].Name());
                                        }
                                    }
                                }

                                for (int i = 0; i < domain.GetLength(1); i++)
                                {
                                    if (tileVars[totalLevel, i].Name().Contains(tensor.Dims[d], StringComparison.CurrentCulture))
                                    {
                                        varNames.Add(tileVars[totalLevel, i].Name());
                                    }
                                }

                                dims.Add(string.Join("*", varNames));

                                // from this buffer reside position to last buffer reside position find the first related loop.
                                var finded = false;
                                for (int nl = level; nl < totalLevel; nl++)
                                {
                                    for (int i = (nl == level) ? (loop + 1) : 0; i < ((nl == lastLevel) ? lastLoop + 1 : domain.GetLength(1)); i++)
                                    {
                                        if (tileVars[nl, i].Name().Contains(tensor.Dims[d], StringComparison.CurrentCulture) && !finded)
                                        {
                                            offsets.Add($"{tensor.Dims[d]}{nl + 1}");
                                            finded = true;
                                        }
                                    }
                                }

                                if (!finded)
                                {
                                    offsets.Add($"0");
                                }
                            }

                            var subTensorName = $"L{sl + 1}_{tensor.Name}";
                            writer.Write($"{indent}{subTensorName} = sub({lastName}, {string.Join(", ", offsets.Zip(dims).Select(p => p.First + ", " + p.Second))})");
                            tensorResides[tensor.Name].Add((subTensorName, level, loop));
                            writer.WriteLine($" # size: {sol.Value(bufferSizes[a, level, loop].Var())}");
                        }
                    }
                }
            }

            string indent = string.Empty;

            // file header
            writer.WriteLine(@$"import numpy as np

def sub(arr: np.ndarray, *arg):
    if len(arg) == 0:
      return arr
    slices = [slice(arg[i] * arg[i + 1], (arg[i] + 1) * arg[i + 1]) for i in range(0, len(arg), 2)]
    return arr[slices]

A = np.random.rand({dimValues[2]}, {dimValues[0]})
B = np.random.rand({dimValues[0]}, {dimValues[1]})
C = np.zeros([{dimValues[2]}, {dimValues[1]}])
");
            for (int a = 0; a < tensors.Length; a++)
            {
                tensorResides.Add(tensors[a].Name, new() { (tensors[a].Name, -1, -1) });
            }

            for (int i = 0; i < domain.GetLength(1); i++)
            {
                var lhs = new List<string>() { $"P_{domain[totalLevel, i]}", $"T_{domain[totalLevel, i]}" };
                var rhs = new List<long>() { primtives[i], sol.Value(tileVars[totalLevel, i]) };
                for (int l = 0; l < totalLevel; l++)
                {
                    for (int j = 0; j < domain.GetLength(1); j++)
                    {
                        if (domain[l, j] == domain[totalLevel, i])
                        {
                            lhs.Add($"T_{domain[totalLevel, i]}{l + 1}");
                            rhs.Add(sol.Value(tileVars[l, j]));
                        }
                    }
                }

                writer.WriteLine($"{string.Join(",", lhs)} = {string.Join(",", rhs)}");
            }

            for (int l = totalLevel - 1; l >= 0; l--)
            {
                writer.WriteLine($"{indent}# L{l + 1}: ");
                writer.WriteLine($"{indent}# stored buffer size: {sol.Value(levelBufferSizes[l].Var())}");
                writer.WriteLine($"{indent}# data reads: {sol.Value(levelDataReads[l].Var())}");
                writer.WriteLine($"{indent}# data writes: {sol.Value(levelDataWrites[l].Var())}");
                for (int i = domain.GetLength(1) - 1; i >= 0; i--)
                {
                    DisplayAlloc(indent, l, i);
                    writer.WriteLine($"{indent}for {domain[l, i]}{l + 1} in range(T_{domain[l, i]}{l + 1}):");
                    indent += "  ";
                }
            }

            // k,n,m
            var prims = Enumerable.Range(0, domain.GetLength(1)).Select(i => sol.Value(tileVars[totalLevel, i])).ToArray();
            writer.WriteLine($"{indent}c = sub({tensorResides["C"].Last().Item1}, m1, {prims[2] * primtives[2]}, n1, {prims[1] * primtives[1]})");
            writer.WriteLine($"{indent}a = sub({tensorResides["A"].Last().Item1}, m1, {prims[2] * primtives[2]}, k1, {prims[0] * primtives[0]})");
            writer.WriteLine($"{indent}b = sub({tensorResides["B"].Last().Item1}, k1, {prims[0] * primtives[0]}, n1, {prims[1] * primtives[1]})");
            writer.WriteLine($"{indent}c += a @ b");
            writer.WriteLine($"{indent}# total cycles : {sol.ObjectiveValue()}");
            writer.WriteLine($"print(np.allclose(C, A @ B))");

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
