// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System.Collections.Immutable;
using System.Runtime.CompilerServices;
using Nncase.Diagnostics;
using Nncase.IR;
using Nncase.TIR;

namespace Nncase.Passes.Tile;

/// <summary>
/// the multi Fusion checker.
/// </summary>
internal sealed class MultiFusionChecker : IFusionChecker
{
    public bool Check(Fusion fusion, RunPassContext passOptions) => false;

    public PrimFunction Convert(RunPassContext passOptions) => throw new NotImplementedException();
}
