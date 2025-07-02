﻿// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Diagnostics.CodeAnalysis;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading.Tasks;
using DryIoc;
using Microsoft.Extensions.DependencyInjection;
using NetFabric.Hyperlinq;
using Nncase.CostModel;
using Nncase.Diagnostics;
using Nncase.Evaluator;
using Nncase.IR;
using Nncase.IR.Affine;
using Nncase.Passes;
using Nncase.PatternMatch;
using Nncase.Schedule;
using Nncase.Targets;

namespace Nncase;

/// <summary>
/// Compiler services provider.
/// </summary>
public interface ICompilerServicesProvider
{
    /// <summary>
    /// Inference type of the expression tree.
    /// </summary>
    /// <param name="expr">Expression.</param>
    /// <param name="inferencer_cache">Inference cache.</param>
    /// <returns>Is fully inferenced.</returns>
    bool InferenceType(BaseExpr expr, Dictionary<Type, ITypeInferencer> inferencer_cache);

    /// <summary>
    /// Inference operator.
    /// </summary>
    /// <param name="op">Target operator.</param>
    /// <param name="context">Inference context.</param>
    /// <param name="inferencer_cache"> inferencer cache.</param>
    /// <returns>Inference result.</returns>
    IRType InferenceOp(Op op, ITypeInferenceContext context, Dictionary<Type, ITypeInferencer> inferencer_cache);

    string? PrintOp(Op op, IPrintOpContext context);

    string Print(IRType type, Diagnostics.PrinterFlags flags);

    string Print(BaseExpr expr, Diagnostics.PrinterFlags flags);

    /// <summary>
    /// if expr is callable will write to {dumpPath}/{prefix}_{callable.name}.{ext}`
    /// else write to {dumpPath}/{prefix}_{expr.Type.name}.il`.
    /// </summary>
    void DumpIR(BaseExpr expr, string prefix, string dumpPath, Diagnostics.PrinterFlags flags);

    /// <summary>
    /// if expr is callable will write to {dumpPath}/{prefix}_{callable.name}.dot`.
    /// <remarks>
    /// not support prim func/prim func wrapper.
    /// </remarks>
    /// </summary>
    void DumpDotIR(BaseExpr expr, string prefix, string dumpPath, Diagnostics.PrinterFlags flags);

    /// <summary>
    /// dump the expr as csharp code.
    /// </summary>
    /// <param name="expr">expression.</param>
    /// <param name="prefix">file prefix.</param>
    /// <param name="dumpDir">file dump ir.</param>
    /// <param name="randConst">false for save const into bin.</param>
    void DumpCSharpIR(BaseExpr expr, string prefix, string dumpDir, bool randConst);

    /// <summary>
    /// dump the expr as csharp code.
    /// </summary>
    /// <param name="expr">expression.</param>
    /// <param name="prefix">file prefix.</param>
    /// <param name="dumpDir">file dump ir.</param>
    void DumpPatternIR(BaseExpr expr, string prefix, string dumpDir);

    /// <summary>
    /// Evaluate the expression tree.
    /// </summary>
    /// <param name="expr">Expression.</param>
    /// <param name="varsValues">Optional vars' values.</param>
    /// <param name="evaluator_cache"> Optional evaluator cache. </param>
    /// <returns>Evaluate result.</returns>
    IValue Evaluate(BaseExpr expr, IReadOnlyDictionary<IVar, IValue>? varsValues = null, Dictionary<Type, IEvaluator>? evaluator_cache = null);

    /// <summary>
    /// Evaluate operator.
    /// </summary>
    /// <param name="op">Target operator.</param>
    /// <param name="context">Evaluate context.</param>
    /// <param name="evaluator_cache"> Optional evaluator cache. </param>
    /// <returns>Evaluate result.</returns>
    IValue EvaluateOp(Op op, IEvaluateContext context, Dictionary<Type, IEvaluator>? evaluator_cache = null);

    /// <summary>
    /// Evaluate cost of the expression tree.
    /// </summary>
    /// <param name="expr">Expression.</param>
    /// <param name="compileOptions">options.</param>
    /// <returns>Evaluate result.</returns>
    Cost EvaluateCost(BaseExpr expr, CompileOptions compileOptions);

    /// <summary>
    /// Evaluate metric of the expression tree.
    /// </summary>
    /// <param name="expr">Expression.</param>
    /// <returns>Evaluate result.</returns>
    Dictionary<BaseExpr, Metric> EvaluateMetric(BaseExpr expr);

    /// <summary>
    /// Evaluate cost of operator.
    /// </summary>
    /// <param name="op">Target operator.</param>
    /// <param name="context">Evaluate context.</param>
    /// <returns>Evaluate result.</returns>
    Cost EvaluateOpCost(Op op, ICostEvaluateContext context);

    /// <summary>
    /// Evaluate metric of operator.
    /// </summary>
    /// <param name="op">Target operator.</param>
    /// <param name="context">Evaluate context.</param>
    /// <returns>Evaluate result.</returns>
    Metric EvaluateOpMetric(Op op, IMetricEvaluateContext context);

    /// <summary>
    /// Match expression.
    /// </summary>
    /// <param name="expr">Expression to match.</param>
    /// <param name="pattern">Match pattern.</param>
    /// <param name="options">Match options.</param>
    /// <param name="result">Match result.</param>
    /// <returns>Match success.</returns>
    bool TryMatch(BaseExpr expr, IPattern pattern, MatchOptions options, [MaybeNullWhen(false)] out IMatchResult result);

    /// <summary>
    /// Match expression as root.
    /// </summary>
    /// <param name="expr">Expression to match.</param>
    /// <param name="pattern">Match pattern.</param>
    /// <param name="options">Match options.</param>
    /// <param name="result">Match result.</param>
    /// <returns>Match success.</returns>
    bool TryMatchRoot(BaseExpr expr, IPattern pattern, MatchOptions options, [MaybeNullWhen(false)] out IMatchResult result);

    /// <summary>
    /// Rewrite expression.
    /// </summary>
    /// <param name="expr">Expression.</param>
    /// <param name="rules">Rewrite rules.</param>
    /// <param name="options">Options.</param>
    /// <returns>Rewrited expression.</returns>
    BaseExpr Rewrite(BaseExpr expr, IEnumerable<IRewriteRule> rules, RunPassContext options);

    /// <summary>
    /// Match enodes as root.
    /// </summary>
    /// <param name="enodes">ENodes.</param>
    /// <param name="pattern">Pattern.</param>
    /// <param name="results">Match results.</param>
    /// <returns>Match success.</returns>
    bool TryMatchRoot(IEnumerable<ENode> enodes, IPattern pattern, [MaybeNullWhen(false)] out IReadOnlyList<IMatchResult> results);

    /// <summary>
    /// Egraph Match Expr as root.
    /// </summary>
    /// <param name="expr">Expression.</param>
    /// <param name="pattern">Pattern.</param>
    /// <param name="results">Match results.</param>
    /// <returns>Match success.</returns>
    bool TryEMatchRoot(BaseExpr expr, IPattern pattern, [MaybeNullWhen(false)] out IReadOnlyList<IMatchResult> results);

    /// <summary>
    /// Get target.
    /// </summary>
    /// <param name="name">Target name.</param>
    /// <returns>Target.</returns>
    ITarget GetTarget(string name);

    /// <summary>
    /// Using EGraph rewrite expression.
    /// </summary>
    /// <param name="expr">Expression.</param>
    /// <param name="rules">Rewrite rules.</param>
    /// <param name="options">Options.</param>
    /// <param name="compileOptions">compileOptions.</param>
    /// <returns>Rewrited expression.</returns>
    BaseExpr ERewrite(BaseExpr expr, IEnumerable<IRewriteRule> rules, RunPassContext options, CompileOptions compileOptions);

    /// <summary>
    /// Using EGraph rewrite expression.
    /// </summary>
    /// <param name="expr">Expression.</param>
    /// <param name="rules">Rewrite rules.</param>
    /// <param name="options">Options.</param>
    /// <returns>Rewrited expression.</returns>
    IEGraph ERewrite(IEGraph expr, IEnumerable<IRewriteRule> rules, RunPassContext options);

    MicroKernelInfo GetOpMicroKernelInfo(Op op, MicroKernelContext context);

    Expr SimplifyForDimension(Expr value);

    bool TryGetMaxShape(Shape shape, [MaybeNullWhen(false)] out long[] maxShape);
}

internal interface ICompilerServicesProviderInternal
{
    IDataTypeServiceProvider DataTypeService { get; }
}

/// <summary>
/// Compiler services.
/// </summary>
public static class CompilerServices
{
    private static IServiceProvider? _serviceProvider;
    private static ICompilerServicesProvider? _provider;

    /// <summary>
    /// Gets root services.
    /// </summary>
    internal static IServiceProvider ServiceProvider => _serviceProvider ?? throw new InvalidOperationException("Compiler services provider must be set.");

    internal static IDataTypeServiceProvider DataTypeService => ((ICompilerServicesProviderInternal)Provider).DataTypeService;

    private static ICompilerServicesProvider Provider => _provider ?? throw new InvalidOperationException("Compiler services provider must be set.");

    /// <summary>
    /// Configure compiler services.
    /// </summary>
    /// <param name="serviceProvider">Root service provider.</param>
    public static void Configure(IServiceProvider serviceProvider)
    {
        _serviceProvider = serviceProvider;
        _provider = serviceProvider.GetRequiredService<ICompilerServicesProvider>();
    }

    public static void Convert<TFrom, TTo>(ReadOnlySpan<TFrom> source, Span<TTo> dest, CastMode castMode = CastMode.KDefault)
        where TFrom : unmanaged, IEquatable<TFrom>
        where TTo : unmanaged, IEquatable<TTo>
    {
        if (typeof(TFrom) == typeof(TTo))
        {
            var sourceSpan = MemoryMarshal.Cast<TFrom, TTo>(source);
            sourceSpan.CopyTo(dest);
        }
        else
        {
            var converter = DataTypeService.GetConverter<TFrom, TTo>();
            converter.ConvertTo(source, dest, castMode);
        }
    }

    /// <summary>
    /// Inference type of the expression tree.
    /// </summary>
    /// <param name="expr">Expression.</param>
    /// <param name="inferencer_cache">Inference cache.</param>
    /// <returns>Is fully inferenced.</returns>
    public static bool InferenceType(this BaseExpr expr, Dictionary<Type, ITypeInferencer> inferencer_cache = null!)
    {
        return Provider.InferenceType(expr, inferencer_cache);
    }

    /// <summary>
    /// Inference operator.
    /// </summary>
    /// <param name="op">Target operator.</param>
    /// <param name="context">Inference context.</param>
    /// <param name="inferencer_cache"> inferencer cache.</param>
    /// <returns>Inference result.</returns>
    public static IRType InferenceOp(Op op, ITypeInferenceContext context, Dictionary<Type, ITypeInferencer> inferencer_cache)
    {
        return Provider.InferenceOp(op, context, inferencer_cache);
    }

    /// <summary>
    /// Evaluate the expression tree.
    /// todo need a method can hook the temporary value.
    /// </summary>
    /// <param name="expr">Expression.</param>
    /// <param name="varsValues">Optional vars' values.</param>
    /// <param name="evaluator_cache"> Optional evaluator cache. </param>
    /// <returns>Evaluate result.</returns>
    public static IValue Evaluate(this BaseExpr expr, IReadOnlyDictionary<IVar, IValue>? varsValues = null, Dictionary<Type, IEvaluator>? evaluator_cache = null)
    {
        return Provider.Evaluate(expr, varsValues, evaluator_cache);
    }

    /// <summary>
    /// Evaluate operator.
    /// </summary>
    /// <param name="op">Target operator.</param>
    /// <param name="context">Evaluate context.</param>
    /// <param name="evaluator_cache"> Optional evaluator cache. </param>
    /// <returns>Evaluate result.</returns>
    public static IValue EvaluateOp(Op op, IEvaluateContext context, Dictionary<Type, IEvaluator>? evaluator_cache = null)
    {
        return Provider.EvaluateOp(op, context, evaluator_cache);
    }

    /// <summary>
    /// Evaluate cost of the expression tree.
    /// </summary>
    /// <param name="expr">Expression.</param>
    /// <param name="compileOptions">compileOptions.</param>
    /// <returns>Evaluate result.</returns>
    public static Cost EvaluateCost(BaseExpr expr, CompileOptions compileOptions)
    {
        return Provider.EvaluateCost(expr, compileOptions);
    }

    /// <summary>
    /// Evaluate operator metric.
    /// </summary>
    /// <param name="op">Target operator.</param>
    /// <param name="context">Evaluate context.</param>
    /// <returns>Evaluate result.</returns>
    public static Metric EvaluateOpMetric(Op op, Evaluator.IMetricEvaluateContext context) => Provider.EvaluateOpMetric(op, context);

    /// <summary>
    /// Evaluate cost of the expression tree.
    /// </summary>
    /// <param name="expr">Expression.</param>
    /// <returns>Evaluate result.</returns>
    public static Dictionary<BaseExpr, Metric> EvaluateMetric(BaseExpr expr) => Provider.EvaluateMetric(expr);

    public static MicroKernelInfo GetOpMicroKernelInfo(Op op, MicroKernelContext context) => Provider.GetOpMicroKernelInfo(op, context);

    /// <summary>
    /// Evaluate cost of operator.
    /// </summary>
    /// <param name="op">Target operator.</param>
    /// <param name="context">Evaluate context.</param>
    /// <returns>Evaluate result.</returns>
    public static Cost EvaluateOpCost(Op op, ICostEvaluateContext context)
    {
        return Provider.EvaluateOpCost(op, context);
    }

    /// <summary>
    /// Match expression.
    /// </summary>
    /// <param name="expr">Expression to match.</param>
    /// <param name="pattern">Match pattern.</param>
    /// <param name="result">Match result.</param>
    /// <returns>Match success.</returns>
    public static bool TryMatch(Expr expr, IPattern pattern, [MaybeNullWhen(false)] out IMatchResult result)
    {
        return Provider.TryMatch(expr, pattern, new MatchOptions(), out result);
    }

    /// <summary>
    /// Match expression.
    /// </summary>
    /// <param name="expr">Expression to match.</param>
    /// <param name="pattern">Match pattern.</param>
    /// <param name="options">Match options.</param>
    /// <param name="result">Match result.</param>
    /// <returns>Match success.</returns>
    public static bool TryMatch(Expr expr, IPattern pattern, MatchOptions options, [MaybeNullWhen(false)] out IMatchResult result)
    {
        return Provider.TryMatch(expr, pattern, options, out result);
    }

    /// <summary>
    /// Match expression as root.
    /// </summary>
    /// <param name="expr">Expression to match.</param>
    /// <param name="pattern">Match pattern.</param>
    /// <param name="result">Match result.</param>
    /// <returns>Match success.</returns>
    public static bool TryMatchRoot(BaseExpr expr, IPattern pattern, [MaybeNullWhen(false)] out IMatchResult result)
    {
        return Provider.TryMatchRoot(expr, pattern, new MatchOptions(), out result);
    }

    /// <summary>
    /// Match expression as root.
    /// </summary>
    /// <param name="expr">Expression to match.</param>
    /// <param name="pattern">Match pattern.</param>
    /// <param name="options">Match options.</param>
    /// <param name="result">Match result.</param>
    /// <returns>Match success.</returns>
    public static bool TryMatchRoot(BaseExpr expr, IPattern pattern, MatchOptions options, [MaybeNullWhen(false)] out IMatchResult result)
    {
        return Provider.TryMatchRoot(expr, pattern, options, out result);
    }

    /// <summary>
    /// Rewrite expression.
    /// </summary>
    /// <param name="expr">Expression.</param>
    /// <param name="rules">Rewrite rules.</param>
    /// <param name="options">Options.</param>
    /// <returns>Rewrited expression.</returns>
    public static BaseExpr Rewrite(BaseExpr expr, IEnumerable<IRewriteRule> rules, RunPassContext options)
    {
        return Provider.Rewrite(expr, rules, options);
    }

    /// <summary>
    /// Using EGraph rewrite expression.
    /// </summary>
    /// <param name="expr">Expression.</param>
    /// <param name="rules">Rewrite rules.</param>
    /// <param name="options">Options.</param>
    /// <param name="compileOptions">compileOptions.</param>
    /// <returns>Rewrited expression.</returns>
    public static BaseExpr ERewrite(BaseExpr expr, IEnumerable<IRewriteRule> rules, RunPassContext options, CompileOptions compileOptions)
    {
        return Provider.ERewrite(expr, rules, options, compileOptions);
    }

    /// <summary>
    /// Using EGraph rewrite expression.
    /// </summary>
    /// <param name="graph">Expression.</param>
    /// <param name="rules">Rewrite rules.</param>
    /// <param name="options">Options.</param>
    /// <returns>Rewrited expression.</returns>
    public static IEGraph ERewrite(IEGraph graph, IEnumerable<IRewriteRule> rules, RunPassContext options)
    {
        return Provider.ERewrite(graph, rules, options);
    }

    /// <summary>
    /// Match enodes as root.
    /// </summary>
    /// <param name="enodes">ENodes.</param>
    /// <param name="pattern">Pattern.</param>
    /// <param name="results">Match results.</param>
    /// <returns>Match success.</returns>
    public static bool TryMatchRoot(IEnumerable<ENode> enodes, IPattern pattern, [MaybeNullWhen(false)] out IReadOnlyList<IMatchResult> results)
    {
        return Provider.TryMatchRoot(enodes, pattern, out results);
    }

    /// <summary>
    /// Match enode as root.
    /// </summary>
    /// <param name="enode">ENode.</param>
    /// <param name="pattern">Pattern.</param>
    /// <param name="results">Match results.</param>
    /// <returns>Match success.</returns>
    public static bool TryMatchRoot(ENode enode, IPattern pattern, [MaybeNullWhen(false)] out IReadOnlyList<IMatchResult> results)
    {
        return Provider.TryMatchRoot(new[] { enode }, pattern, out results);
    }

    /// <summary>
    /// Egraph Match Expr as root.
    /// </summary>
    /// <param name="expr">Expression.</param>
    /// <param name="pattern">Pattern.</param>
    /// <param name="results">Match results.</param>
    /// <returns>Match success.</returns>
    public static bool TryEMatchRoot(Expr expr, IPattern pattern, [MaybeNullWhen(false)] out IReadOnlyList<IMatchResult> results)
    {
        return Provider.TryEMatchRoot(expr, pattern, out results);
    }

    /// <summary>
    /// printer op.
    /// </summary>
    /// <param name="op">Target operator.</param>
    /// <param name="context">Context.</param>
    /// <returns>Result.</returns>
    public static string? PrintOp(Op op, IPrintOpContext context) => Provider.PrintOp(op, context);

    public static void DumpIR(BaseExpr expr, string prefix, string dumpPath, Diagnostics.PrinterFlags flags = Diagnostics.PrinterFlags.Normal) =>
      Provider.DumpIR(expr, prefix, dumpPath, flags);

    /// <summary>
    /// if expr is callable will write to {dumpPath}/{prefix}_{callable.name}.dot`.
    /// <remarks>
    /// not support prim func/prim func wrapper.
    /// </remarks>
    /// </summary>
    public static void DumpDotIR(BaseExpr expr, string prefix, string dumpPath, Diagnostics.PrinterFlags flags = Diagnostics.PrinterFlags.Normal) =>
      Provider.DumpDotIR(expr, prefix, dumpPath, flags);

    /// <summary>
    /// dump the expr as csharp code.
    /// </summary>
    /// <param name="expr">expression.</param>
    /// <param name="prefix">file prefix.</param>
    /// <param name="dumpDir">file dump ir.</param>
    /// <param name="randConst">randConst = false will save the const into bin.</param>
    public static void DumpCSharpIR(BaseExpr expr, string prefix, string dumpDir, bool randConst = true) =>
      Provider.DumpCSharpIR(expr, prefix, dumpDir, randConst);

    /// <summary>
    /// dump the expr as csharp code.
    /// </summary>
    /// <param name="expr">expression.</param>
    /// <param name="prefix">file prefix.</param>
    /// <param name="dumpDir">file dump ir.</param>
    public static void DumpPatternIR(BaseExpr expr, string prefix, string dumpDir) =>
      Provider.DumpPatternIR(expr, prefix, dumpDir);

    public static string Print(IRType type, Diagnostics.PrinterFlags flags = Diagnostics.PrinterFlags.Minimal | Diagnostics.PrinterFlags.SkipDimensionExpr) => Provider.Print(type, flags);

    public static string Print(BaseExpr expr, Diagnostics.PrinterFlags flags = Diagnostics.PrinterFlags.Minimal | Diagnostics.PrinterFlags.SkipDimensionExpr) => Provider.Print(expr, flags);

    /// <summary>
    /// Get target.
    /// </summary>
    /// <param name="name">Target name.</param>
    /// <returns>Target.</returns>
    public static ITarget GetTarget(string name) => Provider.GetTarget(name);

    public static Expr SimplifyForDimension(Expr value) => Provider.SimplifyForDimension(value);

    public static bool TryGetMaxShape(Shape shape, [MaybeNullWhen(false)] out long[] maxShape) => Provider.TryGetMaxShape(shape, out maxShape);

    public static long[] GetMaxShape(Shape shape)
    {
        if (TryGetMaxShape(shape, out var maxShape))
        {
            return maxShape;
        }

        throw new InvalidOperationException("Failed to get max shape.");
    }

    public static Expr FastSimplifyForDimension(Expr value)
    {
        if (value is TensorConst tc)
        {
            return tc.Value.ElementType == DataTypes.Int64 ? tc : new TensorConst(tc.Value.Cast<long>());
        }
        else if (value is None)
        {
            return value;
        }
        else if (value is Var)
        {
            return value.CheckedType is TensorType tt && tt.DType == DataTypes.Int64 ? value : IR.F.Tensors.Cast(value, DataTypes.Int64);
        }
        else if ((value.CheckedType is TensorType tt && tt.DType != DataTypes.Int64)
                || (value.CheckedType is DistributedType dt && dt.TensorType.DType != DataTypes.Int64))
        {
            return SimplifyForDimension(IR.F.Tensors.Cast(value, DataTypes.Int64));
        }
        else if ((value is Call call && call.Arguments.AsValueEnumerable().All(x => x is Const))
            || value.CheckedType is DistributedType)
        {
            return SimplifyForDimension(value);
        }

        return value;
    }

    internal static DryIoc.IContainer CreateScope()
    {
        var container = (DryIoc.IContainer)_serviceProvider!;
        var childDefaultServiceKey = new object();
        var rules = container.Rules
            .WithDefaultRegistrationServiceKey(childDefaultServiceKey)
            .WithFactorySelector(Rules.SelectKeyedOverDefaultFactory(childDefaultServiceKey));
        return container!.With(
            container.Parent,
            rules,
            container.ScopeContext,
            RegistrySharing.CloneButKeepCache,
            container.SingletonScope.Clone(false),
            Scope.Of(container.OwnCurrentScope));
    }
}

internal class CompilerServicesProvider : ICompilerServicesProvider, ICompilerServicesProviderInternal
{
    private readonly IEvaluateProvider _evaluateProvider;
    private readonly ITypeInferenceProvider _typeInferenceProvider;
    private readonly IPrinterProvider _irprinterProvider;
    private readonly ICostEvaluateProvider _costEvaluateProvider;
    private readonly IMetricEvaluateProvider _metricEvaluateProvider;
    private readonly IMatchProvider _matchProvider;
    private readonly IRewriteProvider _rewriteProvider;
    private readonly ISimplifyProvider _simplifyProvider;
    private readonly IEGraphMatchProvider _eGraphMatchProvider;
    private readonly IEGraphRewriteProvider _eGraphrewriteProvider;
    private readonly ITargetProvider _targetProvider;
    private readonly IMicroKernelInfoProvider _microKernelInfoGetter;

    public CompilerServicesProvider(
        IEvaluateProvider evaluateProvider,
        ITypeInferenceProvider typeInferenceProvider,
        IPrinterProvider irprinterProvider,
        ICostEvaluateProvider costEvaluateProvider,
        IMetricEvaluateProvider metricEvaluateProvider,
        IDataTypeServiceProvider dataTypeServiceProvider,
        IMatchProvider matchProvider,
        IRewriteProvider rewriteProvider,
        ISimplifyProvider simplifyProvider,
        IEGraphMatchProvider eGraphMatchProvider,
        IEGraphRewriteProvider eGraphrewriteProvider,
        ITargetProvider targetProvider,
        IMicroKernelInfoProvider microKernelInfoGetter)
    {
        // _compileOptions = compileOptions.Value;
        _evaluateProvider = evaluateProvider;
        _typeInferenceProvider = typeInferenceProvider;
        _irprinterProvider = irprinterProvider;
        _costEvaluateProvider = costEvaluateProvider;
        _metricEvaluateProvider = metricEvaluateProvider;
        DataTypeService = dataTypeServiceProvider;
        _matchProvider = matchProvider;
        _rewriteProvider = rewriteProvider;
        _simplifyProvider = simplifyProvider;
        _eGraphMatchProvider = eGraphMatchProvider;
        _eGraphrewriteProvider = eGraphrewriteProvider;
        _targetProvider = targetProvider;
        _microKernelInfoGetter = microKernelInfoGetter;
    }

    public IDataTypeServiceProvider DataTypeService { get; }

    /// <inheritdoc/>
    public IValue Evaluate(BaseExpr expr, IReadOnlyDictionary<IVar, IValue>? varsValues = null, Dictionary<Type, IEvaluator>? evaluator_cache = null)
    {
        return _evaluateProvider.Evaluate(expr, varsValues, evaluator_cache);
    }

    /// <inheritdoc/>
    public IValue EvaluateOp(Op op, IEvaluateContext context, Dictionary<Type, IEvaluator>? evaluator_cache = null)
    {
        return _evaluateProvider.EvaluateOp(op, context, evaluator_cache);
    }

    /// <inheritdoc/>
    public IRType InferenceOp(Op op, ITypeInferenceContext context, Dictionary<Type, ITypeInferencer> inferencer_cache)
    {
        return _typeInferenceProvider.InferenceOp(op, context, inferencer_cache);
    }

    /// <inheritdoc/>
    public bool InferenceType(BaseExpr expr, Dictionary<Type, ITypeInferencer> inferencer_cache)
    {
        return _typeInferenceProvider.InferenceType(expr, inferencer_cache);
    }

    /// <inheritdoc/>
    public string? PrintOp(Op op, IPrintOpContext context) => _irprinterProvider.PrintOp(op, context);

    /// <inheritdoc/>
    public string Print(IRType type, PrinterFlags flags) => _irprinterProvider.Print(type, flags);

    /// <inheritdoc/>
    public string Print(BaseExpr expr, PrinterFlags flags) => _irprinterProvider.Print(expr, flags);

    /// <inheritdoc/>
    public void DumpIR(BaseExpr expr, string prefix, string dumpPath, Diagnostics.PrinterFlags flags) =>
      _irprinterProvider.DumpIR(expr, prefix, dumpPath, flags);

    /// <inheritdoc/>
    public void DumpDotIR(BaseExpr expr, string prefix, string dumpPath, Diagnostics.PrinterFlags flags) =>
    _irprinterProvider.DumpDotIR(expr, prefix, dumpPath, flags);

    /// <inheritdoc/>
    public void DumpCSharpIR(BaseExpr expr, string prefix, string dumpDir, bool randConst) =>
    _irprinterProvider.DumpCSharpIR(expr, prefix, dumpDir, randConst);

    /// <inheritdoc/>
    public void DumpPatternIR(BaseExpr expr, string prefix, string dumpDir) =>
    _irprinterProvider.DumpPatternIR(expr, prefix, dumpDir);

    /// <inheritdoc/>
    public bool TryMatch(BaseExpr expr, IPattern pattern, MatchOptions options, [MaybeNullWhen(false)] out IMatchResult result)
    {
        return _matchProvider.TryMatch(expr, pattern, options, out result);
    }

    /// <inheritdoc/>
    public bool TryMatchRoot(BaseExpr expr, IPattern pattern, MatchOptions options, [MaybeNullWhen(false)] out IMatchResult result)
    {
        return _matchProvider.TryMatchRoot(expr, pattern, options, out result);
    }

    /// <inheritdoc/>
    public BaseExpr Rewrite(BaseExpr expr, IEnumerable<IRewriteRule> rules, RunPassContext options)
    {
        return _rewriteProvider.Rewrite(expr, rules, options);
    }

    /// <inheritdoc/>
    public Cost EvaluateCost(BaseExpr expr, CompileOptions compileOptions)
    {
        return _costEvaluateProvider.EvaluateCost(expr, compileOptions);
    }

    /// <inheritdoc/>
    public Cost EvaluateOpCost(Op op, ICostEvaluateContext context)
    {
        return _costEvaluateProvider.EvaluateOpCost(op, context);
    }

    /// <inheritdoc/>
    public Dictionary<BaseExpr, Metric> EvaluateMetric(BaseExpr expr) => _metricEvaluateProvider.EvaluateMetric(expr);

    /// <inheritdoc/>
    public Metric EvaluateOpMetric(Op op, IMetricEvaluateContext context) => _metricEvaluateProvider.EvaluateOpMetric(op, context);

    public bool TryMatchRoot(IEnumerable<ENode> enodes, IPattern pattern, [MaybeNullWhen(false)] out IReadOnlyList<IMatchResult> results)
    {
        return _eGraphMatchProvider.TryMatchRoot(enodes, pattern, out results);
    }

    public bool TryEMatchRoot(BaseExpr expr, IPattern pattern, [MaybeNullWhen(false)] out IReadOnlyList<IMatchResult> results)
    {
        return _eGraphMatchProvider.TryEMatchRoot(expr, pattern, out results);
    }

    public ITarget GetTarget(string name)
    {
        return _targetProvider.GetTarget(name);
    }

    public BaseExpr ERewrite(BaseExpr expr, IEnumerable<IRewriteRule> rules, RunPassContext options, CompileOptions compileOptions)
    {
        return _eGraphrewriteProvider.ERewrite(expr, rules, options, compileOptions);
    }

    public IEGraph ERewrite(IEGraph graph, IEnumerable<IRewriteRule> rules, RunPassContext options)
    {
        return _eGraphrewriteProvider.ERewrite(graph, rules, options);
    }

    public MicroKernelInfo GetOpMicroKernelInfo(Op op, MicroKernelContext context) => _microKernelInfoGetter.GetInfo(op, context);

    public Expr SimplifyForDimension(Expr value) => _simplifyProvider.SimplifyForDimension(value);

    public bool TryGetMaxShape(Shape shape, [MaybeNullWhen(false)] out long[] maxShape) => _simplifyProvider.TryGetMaxShape(shape, out maxShape);
}
