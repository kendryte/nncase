// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Diagnostics.CodeAnalysis;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using CommonServiceLocator;
using Microsoft.Extensions.DependencyInjection;
using Nncase.IR;
using Nncase.IR.K210;

namespace Nncase;

/// <summary>
/// Compiler services provider.
/// </summary>
public interface IExtCompilerServicesProvider
{
    /// <summary>
    /// bridge the bounds infer op.
    /// </summary>
    /// <param name="ctx"></param>
    /// <param name="target"></param>
    /// <param name="bounds"></param>
    public void BridgeBoundsInferOp(IR.IBridgeBoundsInferContext ctx, IR.Op target, IRArray<TIR.Range> bounds);

    /// <summary>
    /// make the bounds infer graph from the root expression.
    /// </summary>
    /// <param name="root"></param>
    /// <returns></returns>
    public IR.IBoundsInferGraph MakeBoundsInferGraph(IR.Expr root, IR.Expr? leaf);

    /// <summary>
    /// make the bounds infer tile step for search.
    /// </summary>
    /// <param name="context"></param>
    /// <param name="op"></param>
    public void BridgeTileStepOp(IBridgeBoundsInferContext context, Op op);
}

internal sealed class ExtCompilerServicesProvider : IExtCompilerServicesProvider
{
    private readonly IR.IBoundInferencerProvider _boundInferencerProvider;

    public ExtCompilerServicesProvider(IR.IBoundInferencerProvider boundInferencerProvider)
    {
        _boundInferencerProvider = boundInferencerProvider;
    }

    /// <inheritdoc/>
    public IBoundsInferGraph MakeBoundsInferGraph(Expr root, IR.Expr? leaf) => _boundInferencerProvider.MakeBoundsInferGraph(root, leaf);


    /// <inheritdoc/>
    public void BridgeBoundsInferOp(IR.IBridgeBoundsInferContext ctx, IR.Op target, IRArray<TIR.Range> bounds) => _boundInferencerProvider.BridgeBoundsInferOp(ctx, target, bounds);

    public void BridgeTileStepOp(IBridgeBoundsInferContext ctx, Op target) => _boundInferencerProvider.BridgeTileStepOp(ctx, target);
}

/// <summary>
/// Compiler services.
/// </summary>
public static class ExtCompilerServices
{
    private static IExtCompilerServicesProvider? _provider;

    private static IExtCompilerServicesProvider Provider => _provider ?? throw new InvalidOperationException("Compiler services provider must be set.");

    /// <summary>
    /// Configure compiler services.
    /// </summary>
    /// <param name="provider">Service provider.</param>
    public static void Configure(IExtCompilerServicesProvider provider)
    {
        _provider = provider;
    }

    /// <inheritdoc/>
    public static void BridgeBoundsInferOp(IR.IBridgeBoundsInferContext ctx, IR.Op target, IRArray<TIR.Range> bounds) => Provider.BridgeBoundsInferOp(ctx, target, bounds);

    /// <summary>
    /// bridge the tlie step
    /// </summary>
    /// <param name="context"></param>
    /// <param name="op"></param>
    /// <exception cref="NotImplementedException"></exception>
    public static void BridgeTileStepOp(IBridgeBoundsInferContext context, Op op) => Provider.BridgeTileStepOp(context, op);


    /// <summary>
    /// make the bounds infer graph for the root to leaf
    /// </summary>
    /// <param name="root"></param>
    /// <param name="leaf"></param>
    /// <returns></returns>
    public static IBoundsInferGraph MakeBoundsInferGraph(Expr root, Expr? leaf = null) => Provider.MakeBoundsInferGraph(root, leaf);

}
