// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

namespace Nncase.Passes.Distributed;

public record class DistributedScheme(string Version, string Model, DistributedScheme.Node[] Outputs)
{
    public record class Node(string Name, IR.SBP[] NdSBP, int[] Hierarchy, string HierarchyName)
    {
    }
}
