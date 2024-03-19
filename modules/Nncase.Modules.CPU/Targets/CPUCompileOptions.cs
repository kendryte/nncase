// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Nncase.Targets;

public sealed record CPUCompileOptions(string ModelName, bool Packing, int[] TargetTileSize, int[] Hierarchy, string HierarchyNames, int[] HierarchySizes) : ITargetCompileOptions
{
    public static CPUCompileOptions Default { get; } = new(string.Empty, false, Array.Empty<int>(), new[] { 1 }, "b", new[] { 3 * (int)MathF.Pow(2, 20) });
}
