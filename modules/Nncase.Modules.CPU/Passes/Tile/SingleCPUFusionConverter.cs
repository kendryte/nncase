// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using NetFabric.Hyperlinq;
using Nncase.Diagnostics;
using Nncase.IR;
using Nncase.TIR;

namespace Nncase.Passes.Tile;

/// <summary>
/// convert the fusion to prim func.
/// </summary>
internal sealed class SingleCPUFusionConverter
{
    public SingleCPUFusionConverter(TileOptions tileOptions)
    {
        TileOptions = tileOptions;
    }

    public TileOptions TileOptions { get; }

    public TIR.PrimFunction Convert(Fusion fusion)
    {
        // 1. convert to distribute graph
        var distConverter = new AutoDistributedConvertVisitor(TileOptions);
        var body = distConverter.Convert(fusion.Body);
        var newFusion = fusion.With(body: body);
        if (DumpScope.Current.IsEnabled(DumpFlags.Tiling))
        {
            DumpScope.Current.DumpDotIR(newFusion, newFusion.Name, "Distribute");
        }

        // 2. convert new fusion to prim func
        var primBody = new List<Expr>();
        var visitor = new TIRConvertVisitor(primBody);
        visitor.Visit(newFusion);
        return T.PrimFunc(newFusion.Name, newFusion.ModuleKind, visitor.InputBuffers.Concat(visitor.OutputBuffers).ToArray()).Body(primBody.ToArray()).Build();
    }
}
