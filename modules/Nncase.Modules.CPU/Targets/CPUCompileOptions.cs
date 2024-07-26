// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Nncase.Targets;

public enum MemoryArch : byte
{
    /// <summary>
    /// Unified Memory Access.
    /// </summary>
    UMA = 0,

    /// <summary>
    /// Non-Unified Memory Access.
    /// </summary>
    NUMA = 1,
}

public enum NocArch : byte
{
    Mesh = 0,
    CrossBar = 1,
}

public class CpuTargetOptions : ITargetOptions
{
    public string ModelName { get; set; } = string.Empty;

    public bool Packing { get; set; }

    public int[] TargetTileSize { get; set; } = Array.Empty<int>();

    /// <summary>
    /// Gets or sets a value indicating whether Unified Memory Architecture. see https://en.wikipedia.org/wiki/Unified_Memory_Access.
    /// </summary>
    public bool UnifiedMemoryArchitecture { get; set; } = true;

    public MemoryArch MemoryArch { get; set; } = MemoryArch.UMA;

    public NocArch NocArch { get; set; } = NocArch.Mesh;

    public int[][] Hierarchies { get; set; } = new int[][] { new int[] { 1 } };

    public string HierarchyNames { get; set; } = "b";

    public int[] HierarchySizes { get; set; } = new[] { 3 * (int)MathF.Pow(2, 20) };

    public int[] MemoryCapacity { get; set; } = Array.Empty<int>();

    public int[] MemoryBandWidth { get; set; } = Array.Empty<int>();

    public string DistributedScheme { get; set; } = string.Empty;
}
