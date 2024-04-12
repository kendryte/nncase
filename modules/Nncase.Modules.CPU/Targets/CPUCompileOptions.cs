// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Nncase.Targets;

public sealed class CpuTargetOptions : ITargetOptions
{
    public string ModelName { get; set; } = string.Empty;

    public bool Packing { get; set; }

    public int[] TargetTileSize { get; set; } = Array.Empty<int>();

    public int[] Hierarchy { get; set; } = new[] { 1 };

    public string HierarchyNames { get; set; } = "b";

    public int[] HierarchySizes { get; set; } = new[] { 3 * (int)MathF.Pow(2, 20) };
}
