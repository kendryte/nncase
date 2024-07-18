// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Reactive;
using System.Text;
using System.Threading.Tasks;
using CommunityToolkit.HighPerformance;
using Google.OrTools.ConstraintSolver;
using Nncase.IR;
using Nncase.IR.Affine;
using Nncase.Targets;
using Nncase.Utilities;

namespace Nncase.Schedule;

internal sealed class TilingSolver
{
    public TilingSolver(ITargetOptions targetOptions)
    {
        TargetOptions = targetOptions;
    }

    public ITargetOptions TargetOptions { get; }

    public GridSchedule Solve(int[] domainBounds, int[][] bufferShapes, AffineDim[] domain, AffineMap[] accessMaps, Op computation, int elemSize)
    {
        int[] memoryCapacitys = new[] { 512 * 1024, int.MaxValue };
        int[] memoryBandWidths = new[] { 128, 4 };

        // string prefix, long bestObjective
        var defaultPerms = Enumerable.Range(0, domain.Length).ToArray();
        var singleLevelPerms = Enumerable.Repeat(defaultPerms, domain.Length).CartesianProduct().Where(arr => new HashSet<int>(arr).Count == domain.Length).Select(arr => arr.ToArray()).ToArray();
        long bestObjective = long.MaxValue;
        GridSchedule? result = null;
        var loopMasks = accessMaps.Select(GetLoopMasks).ToArray();
        foreach (var multiLevelPerms in Enumerable.Repeat(singleLevelPerms, memoryCapacitys.Length).CartesianProduct())
        {
            // note we revese the loop domain to inner first.
            var newDomain = domain.Reverse().ToArray();
            var perms = multiLevelPerms.ToArray();
            var newFullDomain = new AffineDim[memoryCapacitys.Length + 1, newDomain.Length];
            for (int l = 0; l < memoryCapacitys.Length; l++)
            {
                for (int i = 0; i < newDomain.Length; i++)
                {
                    newFullDomain[l, i] = newDomain[perms[l][i]];
                }
            }

            // but we keep the buffer shapes/domain dounds order.
            for (int i = 0; i < newDomain.Length; i++)
            {
                newFullDomain[memoryCapacitys.Length, i] = domain[i];
            }

            result = SolveWithPermutation(domainBounds, bufferShapes, newFullDomain, accessMaps, loopMasks, memoryCapacitys, memoryBandWidths, $"dd", ref bestObjective, computation, elemSize);
            if (result is not null)
            {
                return result;
            }
        }

        return result!;
    }

    private static LoopMasks GetLoopMasks(AffineMap map)
    {
        var masks = new LoopMask[map.Results.Length];
        for (int i = 0; i < map.Results.Length; i++)
        {
            var dimsCollector = new AffineDimCollector();
            dimsCollector.Visit(map.Results[i]);

            uint mask = 0;
            for (int j = 0; j < map.Domains.Length; j++)
            {
                if (dimsCollector.AffineDims.Contains(map.Domains[j].Offset))
                {
                    mask |= 1U << map.Domains[j].Offset.Position;
                }
            }

            masks[i] = new LoopMask(mask);
        }

        return new(masks);
    }

    private GridSchedule? SolveWithPermutation(int[] domainBounds, int[][] bufferShapes, AffineDim[,] fullDomain, AffineMap[] accessMaps, LoopMasks[] loopMasks, int[] memoryCapacitys, int[] memoryBandWidths, string prefix, ref long bestObjective, Op computation, int elemSize)
    {
        var totalLevel = memoryCapacitys.Length;
        var model = new Solver("tiling");
        IntExpr one = model.MakeIntConst(1, "one");
        IntExpr zero = model.MakeIntConst(0, "zero");
        IntExpr elem = model.MakeIntConst(elemSize, "elem");
        var info = CompilerServices.GetOpMicroKernelInfo(computation, fullDomain.GetRow(totalLevel).ToArray(), accessMaps, bufferShapes, TargetOptions);
        var primitiveSizes = info.Primitives;
        var primitiveMultiplier = info.Multiplier;

        // 1. create tilesize vars, and we save the statement level tile size var in the last row.
        var tileVars = new IntVar[totalLevel + 1, fullDomain.GetLength(1)];
        for (int l = 0; l < totalLevel; l++)
        {
            for (int i = 0; i < fullDomain.GetLength(1); i++)
            {
                for (int j = 0; j < fullDomain.GetLength(1); j++)
                {
                    if (fullDomain[l, i] == fullDomain[totalLevel, j])
                    {
                        tileVars[l, i] = model.MakeIntVar(1, domainBounds[j] / primitiveSizes[j], $"T_{fullDomain[l, i]}_{l + 1}");
                        break;
                    }
                }
            }
        }

        for (int i = 0; i < fullDomain.GetLength(1); i++)
        {
            tileVars[totalLevel, i] = model.MakeIntVar(1, domainBounds[i] / primitiveSizes[i], $"T_{fullDomain[totalLevel, i]}");
            tileVars[totalLevel, i].SetRange(primitiveMultiplier[i].Min, primitiveMultiplier[i].Max);
        }

        // 2. the reads table save the buffer read times, we can use it directly.
        var primitiveBufferShapes = new IntExpr[bufferShapes.Length][];
        var primitiveBufferSizes = new IntExpr[bufferShapes.Length];
        for (int a = 0; a < bufferShapes.Length; a++)
        {
            primitiveBufferSizes[a] = elem;
            primitiveBufferShapes[a] = new IntExpr[bufferShapes[a].Length];
            var extentVars = tileVars.GetRow(totalLevel).ToArray();
            var converter = new AffineExprToIntExprConverter(model, extentVars);
            var primtiveMap = AffineMap.FromCallable((doms, syms) => Enumerable.Range(0, domainBounds.Length).Select(i => new AffineRange(doms[i].Offset, new AffineMulBinary(doms[i].Extent, new AffineConstant(primitiveSizes[i])))).ToArray(), domainBounds.Length);
            var composedMap = primtiveMap * accessMaps[a];
            for (int i = 0; i < bufferShapes[a].Length; i++)
            {
                primitiveBufferShapes[a][i] = converter.Visit(composedMap.Results[i].Extent);
                primitiveBufferSizes[a] *= primitiveBufferShapes[a][i];
            }
        }

        IntExpr totalLoopTimes = Enumerable.Range(0, totalLevel).Select(i => Enumerable.Range(0, fullDomain.GetLength(1)).Select(j => tileVars[i, j])).SelectMany(i => i).Aggregate(one, (acc, tileVar) => acc * tileVar);

        // 3. create buffer placement vars: gates(l,i,sl) mean create buffer size by memory level l, loop i, then store it into memory level sl.
        var placeGates = new IntVar[bufferShapes.Length, totalLevel, fullDomain.GetLength(1)][];
        var dataWrites = new IntExpr[bufferShapes.Length, totalLevel, fullDomain.GetLength(1)][];
        var bufferSizes = new IntExpr[bufferShapes.Length, totalLevel, fullDomain.GetLength(1)];
        for (int ts = 0; ts < bufferShapes.Length; ts++)
        {
            for (int l = 0; l < totalLevel; l++)
            {
                for (int i = 0; i < fullDomain.GetLength(1); i++)
                {
                    {
                        var lastLevelSize = (l == 0 && i == 0) ? primitiveBufferSizes[ts] : (i == 0 ? bufferSizes[ts, l - 1, fullDomain.GetLength(1) - 1] : bufferSizes[ts, l, i - 1]);
                        bufferSizes[ts, l, i] = loopMasks[ts].IsRelated(fullDomain[l, i]) ? lastLevelSize * tileVars[l, i] : lastLevelSize;
                    }

                    // we can create buffer size by high loop level, but put it into low loop level.
                    var subLevelPlace = placeGates[ts, l, i] = new IntVar[l + 1];
                    var subLevelWrites = dataWrites[ts, l, i] = new IntExpr[l + 1];
                    for (int sl = 0; sl < l + 1; sl++)
                    {
                        if (l == 1 && sl == 1)
                        {
                            subLevelPlace[sl] = model.MakeIntConst(0, $"place(b{ts}, {l}, {fullDomain[l, i]}, {sl})");
                        }
                        else
                        {
                            subLevelPlace[sl] = model.MakeBoolVar($"place(b{ts}, {l}, {fullDomain[l, i]}, {sl})");
                        }

                        // 2. compute data writes
                        subLevelWrites[sl] = bufferSizes[ts, l, i];
                        for (int nl = l; nl < totalLevel; nl++)
                        {
                            for (int ci = 0; ci < fullDomain.GetLength(1); ci++)
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

        // 4. in each create level create data reads.
        var dataReads = new IntExpr[bufferShapes.Length, totalLevel];
        for (int a = 0; a < bufferShapes.Length; a++)
        {
            for (int l = 0; l < totalLevel; l++)
            {
                if (l == 0)
                {
                    // note l0 read should compute by computation
                    dataReads[a, l] = zero;
                }
                else
                {
                    dataReads[a, l] = zero;
                    for (int i = 0; i < fullDomain.GetLength(1); i++)
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

        // start add constraints.
        // 1. must store one buffer at lowest level.
        var lowestStoredBufferNums = new IntExpr[bufferShapes.Length];
        for (int a = 0; a < bufferShapes.Length; a++)
        {
            IntExpr lowestBufferNums = zero;
            for (int l = 0; l < totalLevel; l++)
            {
                for (int i = 0; i < fullDomain.GetLength(1); i++)
                {
                    lowestBufferNums += placeGates[a, l, i][0];
                }
            }

            lowestStoredBufferNums[a] = lowestBufferNums;
            var c = model.MakeEquality(lowestStoredBufferNums[a], 1);
            c.SetName($"lowestStoredBufferNums[b{a}]");
            model.Add(c);
        }

        // 2. each tensor only can create one or zero buffer at each create level.
        var eachlevelCreateBufferConstraints = new Constraint[bufferShapes.Length, totalLevel];
        var eachlevelCreateBufferNums = new IntExpr[bufferShapes.Length, totalLevel];
        for (int a = 0; a < bufferShapes.Length; a++)
        {
            for (int l = 0; l < totalLevel; l++)
            {
                IntExpr bufferNums = zero;
                for (int i = 0; i < fullDomain.GetLength(1); i++)
                {
                    var slGates = placeGates[a, l, i];
                    for (int sl = 0; sl < slGates.Length; sl++)
                    {
                        bufferNums += slGates[sl];
                    }
                }

                eachlevelCreateBufferNums[a, l] = bufferNums;
                eachlevelCreateBufferConstraints[a, l] = model.MakeLessOrEqual(bufferNums, one);
                eachlevelCreateBufferConstraints[a, l].SetName($"eachlevelCreateBufferConstraints[b{a}, {l}]");
                model.Add(eachlevelCreateBufferConstraints[a, l]);
            }
        }

        // 3. if current level has create a buffer, it's requires previous level store a buffer.
        var depLevelBufferConstraints = new Constraint[bufferShapes.Length, totalLevel - 1];
        for (int a = 0; a < bufferShapes.Length; a++)
        {
            for (int l = 0; l < totalLevel - 1; l++)
            {
                IntExpr previousLevelStoreBufferNums = zero;
                for (int pl = l + 1; pl < totalLevel; pl++)
                {
                    for (int i = 0; i < fullDomain.GetLength(1); i++)
                    {
                        previousLevelStoreBufferNums += placeGates[a, pl, i][pl];
                    }
                }

                depLevelBufferConstraints[a, l] = model.MakeGreaterOrEqual(previousLevelStoreBufferNums, model.MakeIsEqualVar(eachlevelCreateBufferNums[a, l], one));
                depLevelBufferConstraints[a, l].SetName($"depLevelBufferConstraints[b{a}, {l}]");
                model.Add(depLevelBufferConstraints[a, l]);
            }
        }

        // 4. add tile vars equal to domain value constraints
        var tileVarConstraints = new Constraint[fullDomain.GetLength(1)];
        for (int i = 0; i < fullDomain.GetLength(1); i++)
        {
            IntExpr prod = tileVars[totalLevel, i];

            for (int l = 0; l < totalLevel; l++)
            {
                for (int j = 0; j < fullDomain.GetLength(1); j++)
                {
                    if (fullDomain[l, j] == fullDomain[totalLevel, i])
                    {
                        prod *= tileVars[l, j];
                    }
                }
            }

            tileVarConstraints[i] = model.MakeEquality(prod, domainBounds[i] / primitiveSizes[i]);
            model.Add(tileVarConstraints[i]);
        }

        // 5. add the memory capacity constraints
        var levelBufferSizes = Enumerable.Range(0, totalLevel).Select(i => (IntExpr)model.MakeIntConst(0, "zero")).ToArray();
        for (int a = 0; a < bufferShapes.Length; a++)
        {
            for (int l = 0; l < totalLevel; l++)
            {
                for (int i = 0; i < fullDomain.GetLength(1); i++)
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
        {
            for (int a = 0; a < bufferShapes.Length; a++)
            {
                for (int l = 0; l < totalLevel; l++)
                {
                    levelDataReads[l] += dataReads[a, l];
                    for (int i = 0; i < fullDomain.GetLength(1); i++)
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
                levelCycles[l] = levelCycles[l].CeilDiv(memoryBandWidths[l]);
            }

            // custom the computation level cycles
            {
                var computationLoad = totalLoopTimes * primitiveBufferSizes.SkipLast(1).Aggregate(zero, (acc, s) => acc + s);
                levelCycles[0] += computationLoad.CeilDiv(info.ReadBandWidth);
                var computationStore = totalLoopTimes * primitiveBufferSizes[^1];
                levelCycles[0] += computationStore.CeilDiv(info.WriteBandWidth);
            }
        }

        var totalCycles = levelCycles[0];
        for (int l = 1; l < totalLevel; l++)
        {
            // todo max or sum?
            totalCycles = model.MakeMax(totalCycles, levelCycles[l]);
        }

        var totalCyclesVar = totalCycles.Var();
        totalCyclesVar.SetRange(1, long.MaxValue / memoryBandWidths[0]); /* avoid crash. */

        var objectiveMonitor = model.MakeMinimize(totalCyclesVar, 1);
        var logger = model.MakeSearchTrace($"{prefix}_tiling:");
        var collector = model.MakeNBestValueSolutionCollector(5, false);
        var searchAbleVars = new List<IntVar>();
        searchAbleVars.AddRange(tileVars.AsSpan().ToArray());
        searchAbleVars.AddRange(placeGates.AsSpan().ToArray().SelectMany(i => i).ToArray());
        collector.Add(searchAbleVars.ToArray());
        collector.Add(lowestStoredBufferNums.Select(c => c.Var()).ToArray());
        collector.Add(eachlevelCreateBufferNums.AsSpan().ToArray().Select(c => c.Var()).ToArray());
        collector.Add(primitiveBufferShapes.SelectMany(i => i).Select(i => i.Var()).ToArray());
        collector.Add(levelBufferSizes.Select(c => c.Var()).ToArray());
        collector.Add(levelDataReads.Select(c => c.Var()).ToArray());
        collector.Add(levelDataWrites.Select(c => c.Var()).ToArray());
        for (int ts = 0; ts < bufferShapes.Length; ts++)
        {
            for (int l = 0; l < totalLevel; l++)
            {
                collector.Add(dataReads[ts, l].Var());
                for (int i = 0; i < fullDomain.GetLength(1); i++)
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
        var decisionBuilder = model.MakeDefaultPhase(searchAbleVars.ToArray());
        var status = model.Solve(decisionBuilder, new SearchMonitor[] { collector, objectiveMonitor, model.MakeTimeLimit(5000) });
        {
            if (!status)
            {
                return null;
            }
        }

        var sol = collector.Solution(collector.SolutionCount() - 1);
        if (sol.ObjectiveValue() < bestObjective)
        {
            var bufferScope = new Dictionary<int, List<(GridSchedule.TemporalBuffer? Parent, int Level, int Loop)>>();
            for (int a = 0; a < bufferShapes.Length; a++)
            {
                bufferScope[a] = new();
            }

            var tileSizes = new long[totalLevel + 1, domainBounds.Length];
            for (int l = 0; l < totalLevel + 1; l++)
            {
                for (int i = 0; i < domainBounds.Length; i++)
                {
                    tileSizes[l, i] = sol.Value(tileVars[l, i]);
                }
            }

            var finalLoops = new GridSchedule.Loop[totalLevel, fullDomain.GetLength(1)];
            AffineMap domainMap;
            {
                var dict = new Dictionary<int, List<int>>();
                var loopStops = tileSizes.GetRow(totalLevel).ToArray();
                for (int i = 0; i < loopStops.Length; i++)
                {
                    loopStops[i] *= primitiveSizes[i];
                }

                var newPosition = 0;
                var newDomains = new List<AffineDomain>();
                for (int l = 0; l < totalLevel; l++)
                {
                    for (int i = 0; i < domainBounds.Length; i++)
                    {
                        for (int j = 0; j < domainBounds.Length; j++)
                        {
                            if (fullDomain[l, i] == fullDomain[totalLevel, j])
                            {
                                var stop = loopStops[j] * tileSizes[l, i];
                                var newDomin = new AffineDomain(new AffineDim(newPosition), new AffineExtent(newPosition));
                                newDomains.Add(newDomin);
                                if (!dict.TryGetValue(fullDomain[l, i].Position, out var newPositions))
                                {
                                    newPositions = new();
                                    dict.Add(fullDomain[l, i].Position, newPositions);
                                }

                                newPositions.Add(newPosition);
                                finalLoops[l, i] = new GridSchedule.Loop(newDomin, stop, loopStops[j], $"{fullDomain[l, i]}_{l + 1}");
                                loopStops[j] = stop;
                                newPosition++;
                                break;
                            }
                        }
                    }
                }

                var newResults = new AffineRange[dict.Keys.Count];
                for (int i = 0; i < dict.Keys.Count; i++)
                {
                    var newPositions = dict[i];
                    newResults[i] = new(newPositions.Skip(1).Aggregate((AffineExpr)newDomains[newPositions[0]].Offset, (acc, j) => acc + newDomains[j].Offset), newPositions.Skip(1).Aggregate((AffineExpr)newDomains[newPositions[0]].Extent, (acc, j) => acc + newDomains[j].Extent));
                }

                domainMap = new AffineMap(newDomains.ToArray(), Array.Empty<AffineSymbol>(), newResults.ToArray());
            }

            var finalPlaces = new GridSchedule.Place[totalLevel, fullDomain.GetLength(1)];
            void CreateTemporalBuffer(int level, int loop)
            {
                var createds = new List<GridSchedule.TemporalBuffer>();
                for (int a = 0; a < bufferShapes.Length; a++)
                {
                    var storeGates = placeGates[a, level, loop];
                    for (int sl = 0; sl < storeGates.Length; sl++)
                    {
                        if (sol.Value(storeGates[sl]) == 1)
                        {
                            var (parent, lastLevel, lastLoop) = bufferScope[a].Count == 0 ? (null, -1, -1) : bufferScope[a].Last();
                            var extents = primitiveBufferShapes[a].Select(i => sol.Value(i.Var())).ToArray();
                            var offsets = new List<AffineExpr>(); /* index start */

                            for (int d = 0; d < bufferShapes[a].Length; d++)
                            {
                                for (int l = 0; l <= level; l++)
                                {
                                    for (int i = 0; i < ((l == level) ? loop + 1 : fullDomain.GetLength(1)); i++)
                                    {
                                        if (loopMasks[a][d].IsRelated(fullDomain[l, i]))
                                        {
                                            extents[d] *= tileSizes[l, i];
                                        }
                                    }
                                }

                                // from this buffer reside position to last buffer reside position collect all related loop vars.
                                var finded = false;
                                AffineExpr offset = null!;
                                for (int nl = level; nl <= ((lastLevel == -1) ? totalLevel - 1 : lastLevel); nl++)
                                {
                                    for (int i = (nl == level) ? (loop + 1) : 0; i < ((nl == lastLevel) ? lastLoop + 1 : fullDomain.GetLength(1)); i++)
                                    {
                                        if (loopMasks[a][d].IsRelated(fullDomain[nl, i]))
                                        {
                                            offset = offset is null ? finalLoops[nl, i].Domain.Offset : new AffineAddBinary(offset, finalLoops[nl, i].Domain.Offset);
                                            finded = true;
                                        }
                                    }
                                }

                                if (!finded)
                                {
                                    offsets.Add(0);
                                }
                                else
                                {
                                    offsets.Add(offset);
                                }
                            }

                            // var subTensorName = $"L{sl + 1}_{tensor.Name}";
                            var subBufferDomains = new List<AffineDomain>();
                            for (int nl = level; nl < totalLevel; nl++)
                            {
                                for (int i = (nl == level) ? (loop + 1) : 0; i < fullDomain.GetLength(1); i++)
                                {
                                    subBufferDomains.Add(finalLoops[nl, i].Domain);
                                }
                            }

                            var subBuffer = new GridSchedule.TemporalBuffer(a, new AffineMap(subBufferDomains.ToArray(), Array.Empty<AffineSymbol>(), offsets.Zip(extents).Select(p => new AffineRange(p.First, p.Second)).ToArray()), parent);
                            bufferScope[a].Add((subBuffer, level, loop));
                            createds.Add(subBuffer);
                        }
                    }
                }

                finalPlaces[level, loop] = new GridSchedule.Place(createds.ToArray());
            }

            // GridSchedule result;
            for (int l = totalLevel - 1; l >= 0; l--)
            {
                for (int i = fullDomain.GetLength(1) - 1; i >= 0; i--)
                {
                    // create sub buffer.
                    CreateTemporalBuffer(l, i);
                }
            }

            // computation
            var finalBodyView = new AffineMap[bufferShapes.Length];
            for (int a = 0; a < bufferShapes.Length; a++)
            {
                // note assume the domain only map to one dimension.
                var extents = primitiveBufferShapes[a].Select(i => sol.Value(i.Var())).ToArray();
                var offsets = new List<AffineExpr>();
                for (int d = 0; d < bufferShapes[a].Length; d++)
                {
                    var offsetVars = new List<AffineDim>();
                    var (_, lastLevel, lastLoop) = bufferScope[a].Last();
                    for (int nl = 0; nl <= lastLevel; nl++)
                    {
                        for (int i = 0; i < ((nl == lastLevel) ? lastLoop + 1 : fullDomain.GetLength(1)); i++)
                        {
                            if (loopMasks[a][d].IsRelated(fullDomain[nl, i]))
                            {
                                offsetVars.Add(finalLoops[nl, i].Domain.Offset);
                            }
                        }
                    }

                    offsets.Add(offsetVars.Count == 0 ? new AffineConstant(0) : offsetVars.Skip(1).Aggregate((AffineExpr)offsetVars[0], (acc, v) => new AffineAddBinary(acc, v)));
                }

                finalBodyView[a] = new(finalLoops.AsSpan().ToArray().Reverse().Select(l => l.Domain).ToArray(), Array.Empty<AffineSymbol>(), offsets.Zip(extents).Select(p => new AffineRange(p.First, p.Second)).ToArray());
            }

            bestObjective = sol.ObjectiveValue();
            return new GridSchedule(domainMap, finalLoops.AsSpan().ToArray().Reverse().ToArray(), finalPlaces.AsSpan().ToArray().Reverse().ToArray(), finalBodyView);
        }

        return null;
    }
}

internal sealed class AffineExprToIntExprConverter : ExprVisitor<IntExpr, Unit>
{
    private readonly Solver _solver;
    private readonly IntVar[] _extents;

    public AffineExprToIntExprConverter(Solver solver, IntVar[] extentVars)
    {
        _solver = solver;
        _extents = extentVars;
    }

    protected override IntExpr VisitLeafAffineExtent(AffineExtent expr)
    {
        return _extents[expr.Position];
    }

    protected override IntExpr VisitLeafAffineConstant(AffineConstant expr) =>
        _solver.MakeIntConst(expr.Value);

    protected override IntExpr VisitLeafAffineAddBinary(AffineAddBinary expr) =>
        ExprMemo[expr.Lhs] + ExprMemo[expr.Rhs];

    protected override IntExpr VisitLeafAffineMulBinary(AffineMulBinary expr) =>
        ExprMemo[expr.Lhs] * ExprMemo[expr.Rhs];

    protected override IntExpr VisitLeafAffineDivBinary(AffineDivBinary expr) =>
        expr.BinaryOp switch
        {
            AffineDivBinaryOp.FloorDiv => _solver.MakeDiv(ExprMemo[expr.Lhs], ExprMemo[expr.Rhs]),
            AffineDivBinaryOp.CeilDiv => ExprMemo[expr.Lhs].CeilDiv(ExprMemo[expr.Rhs]),
            AffineDivBinaryOp.Mod => _solver.MakeModulo(ExprMemo[expr.Lhs], ExprMemo[expr.Rhs]),
            _ => throw new UnreachableException(),
        };
}

internal sealed class AffineDimCollector : ExprWalker
{
    public HashSet<AffineDim> AffineDims { get; } = new(ReferenceEqualityComparer.Instance);

    protected override Unit VisitAffineDim(AffineDim expr)
    {
        AffineDims.Add(expr);
        return default;
    }
}


#if false
public class TilingSolver
{
    // 1. Constants
    private const int L2_SIZE = 1024 * 1024 * 4; // 4MB
    private const int L3_BANDWIDTH = 128; // 128B/cycle
    private const int MMA_PRIM_M = 32;
    private const int MMA_PRIM_N = 32;
    private const int MMA_PRIM_K = 32;
    private const int MMA_PRIM_CYCLES = 8;

    private readonly int[][] _bufferShapes;
    private readonly AffineMap[] _accessMaps;
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

    public TilingSolver(int[] dims, int[][] bufferShapes, AffineMap[] accessMaps)
    {
        // 1. Constants
        _bufferShapes = bufferShapes;
        _accessMaps = accessMaps;
        _loopMasks = accessMaps.Select(GetLoopMask).ToArray();
        _loopsCount = dims.Length;
        _reductionLoopsCount = _loopsCount - _loopMasks[^1].Ones;
        _buffersCount = bufferShapes.Length;

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

    public GridSchedule Solve()
    {
        var objeciveMonitor = _solver.MakeMinimize(_objective, 1);
        var searchLog = _solver.MakeSearchLog(100000, objeciveMonitor);
        var searchLimit = _solver.MakeImprovementLimit(_objective, false, 1, 0, 1, 2);
        var timeLimit = _solver.MakeTimeLimit(50000);

        _solver.Solve(_decisionBuilder, new SearchMonitor[] { objeciveMonitor, searchLimit, timeLimit, searchLog, _solutionCollector });

        if (_solutionCollector.SolutionCount() < 1)
        {
            throw new InvalidOperationException();
        }

        var solution = _solutionCollector.SolutionCount() - 1;

        // Generate schedule
        // 1. Loops
        var loops = new GridSchedule.Loop[_loopsCount];
        for (int loop = 0; loop < loops.Length; loop++)
        {
            var domain = GetLoopDomain(solution, loop);
            var tileSize = (int)_solutionCollector.Value(solution, _tiles[domain]);
            loops[loop] = new GridSchedule.Loop(_accessMaps[0].Domains[domain], tileSize);
        }

        // 2. Places & body buffers
        var buffersByPlace = (from b in Enumerable.Range(0, _buffersCount)
                              let place = GetBufferPlace(solution, b)
                              group b by place).ToDictionary(x => x.Key, x => x.ToArray());
        var places = new GridSchedule.Place[_loopsCount + 1];
        var bodyBufferViews = new AffineMap[_buffersCount];
        for (int place = 0; place < places.Length; place++)
        {
            var bufferIds = buffersByPlace.GetValueOrDefault(place, Array.Empty<int>());
            var buffers = new GridSchedule.TemporalBuffer[bufferIds.Length];
            for (int i = 0; i < buffers.Length; i++)
            {
                var buffer = bufferIds[i];
                (var subview, var bodyBufferView) = GetBufferSubview(solution, buffer, place);
                buffers[i] = new GridSchedule.TemporalBuffer(buffer, subview);
                bodyBufferViews[buffer] = bodyBufferView;
            }

            places[place] = new(buffers);
        }

        return new GridSchedule(loops, places, bodyBufferViews);
    }

    private static IntExpr GetTileCalcCycles(IntVar[] tiles)
    {
        return tiles.Aggregate<IntExpr>((x, y) => x * y);
    }

    private static LoopMask GetLoopMask(AffineMap map)
    {
        var dimsCollector = new AffineDimCollector();
        foreach (var result in map.Results)
        {
            dimsCollector.Visit(result);
        }

        uint mask = 0;
        for (int i = 0; i < map.Domains.Length; i++)
        {
            if (dimsCollector.AffineDims.Contains(map.Domains[i].Offset))
            {
                mask |= 1U << i;
            }
        }

        return new LoopMask(mask);
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
                orders[i, j] = _solver.MakeBoolVar($"order_l{i}_d{j}");
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

    private int GetLoopDomain(int solution, int loop)
    {
        for (int domain = 0; domain < _loopsCount; domain++)
        {
            if (_solutionCollector.Value(solution, _orders[loop, domain]) == 1)
            {
                return domain;
            }
        }

        throw new InvalidOperationException();
    }

    private int GetDomainLoop(int solution, int domain)
    {
        for (int i = 0; i < _loopsCount; i++)
        {
            if (_solutionCollector.Value(solution, _orders[i, domain]) == 1)
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

    private (AffineMap SubView, AffineMap BodyView) GetBufferSubview(int solution, int buffer, int place)
    {
        var accessMap = _accessMaps[buffer];
        var placeMask = GetPlaceLoopMask(solution, place);
        var subviewReplaceMap = new Dictionary<AffineExpr, AffineExpr>();
        var bodyViewReplaceMap = new Dictionary<AffineExpr, AffineExpr>();
        for (int domain = 0; domain < _loopsCount; domain++)
        {
            if (!placeMask.IsRelated(domain))
            {
                subviewReplaceMap.Add(accessMap.Domains[domain].Offset, 0);
                subviewReplaceMap.Add(accessMap.Domains[domain].Extent, ((IntVar)_dims[domain]).Value());
            }
            else
            {
                bodyViewReplaceMap.Add(accessMap.Domains[domain].Offset, 0);
            }
        }

        var subviewResults = new AffineRange[accessMap.Results.Length];
        var bodyViewResults = new AffineRange[accessMap.Results.Length];
        {
            var generator = new BufferSubviewGenerator(subviewReplaceMap);
            for (int i = 0; i < subviewResults.Length; i++)
            {
                subviewResults[i] = generator.Clone(accessMap.Results[i], default);
            }
        }

        {
            var generator = new BufferSubviewGenerator(bodyViewReplaceMap);
            for (int i = 0; i < subviewResults.Length; i++)
            {
                bodyViewResults[i] = generator.Clone(accessMap.Results[i], default);
            }
        }

        return (accessMap.With(results: subviewResults), accessMap.With(results: bodyViewResults));
    }

    private LoopMask GetPlaceLoopMask(int solution, int place)
    {
        uint mask = 0;
        for (int i = 1; i <= place; i++)
        {
            var loop = i - 1;
            var domain = GetLoopDomain(solution, loop);
            mask |= 1U << domain;
        }

        return new LoopMask(mask);
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

            if (anyOrder != null)
            {
                var constraint = placeVar * (anyOrder ?? _solver.MakeIntConst(1)) == 0;
                _solver.Add(constraint);
            }
        }
    }

    private record struct OrderCombination(LoopMask Loops)
    {
        public IntExpr Expr { get; set; }
    }

    private sealed class AffineDimCollector : ExprWalker
    {
        public HashSet<AffineDim> AffineDims { get; } = new(ReferenceEqualityComparer.Instance);

        protected override Unit VisitAffineDim(AffineDim expr)
        {
            AffineDims.Add(expr);
            return default;
        }
    }

    private sealed class BufferSubviewGenerator : ExprCloner<Unit>
    {
        private readonly Dictionary<AffineExpr, AffineExpr> _mapper;

        public BufferSubviewGenerator(Dictionary<AffineExpr, AffineExpr> mapper)
        {
            _mapper = mapper;
        }

        protected override Expr VisitAffineDim(AffineDim expr, Unit context)
        {
            if (_mapper.TryGetValue(expr, out var newExpr))
            {
                return newExpr;
            }

            return expr;
        }

        protected override Expr VisitLeafAffineExtent(AffineExtent expr, Unit context)
        {
            if (_mapper.TryGetValue(expr, out var newExpr))
            {
                return newExpr;
            }

            return expr;
        }
    }
}

#endif
