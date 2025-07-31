// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

namespace Nncase.Passes.Distributed;

public record class CustomOpScheme(string Version, string Model, CustomOpScheme.Node[] Outputs)
{
    public record class Node(string? Name, string Op, long[][] Shape, IR.SBP[][] SBP, ulong Cost, string CSourcePath, string FuncName)
    {
    }
}
