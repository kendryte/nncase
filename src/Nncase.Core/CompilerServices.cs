// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Diagnostics.CodeAnalysis;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using DryIoc;
using Microsoft.Extensions.DependencyInjection;
using Nncase.CostModel;
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
    /// <returns>Is fully inferenced.</returns>
    bool InferenceType(Expr expr);

    /// <summary>
    /// Inference operator.
    /// </summary>
    /// <param name="op">Target operator.</param>
    /// <param name="context">Inference context.</param>
    /// <param name="inferencer_cache"> inferencer cache.</param>
    /// <returns>Inference result.</returns>
    IRType InferenceOp(Op op, ITypeInferenceContext context, Dictionary<Type, ITypeInferencer> inferencer_cache);

    /// <summary>
    /// printer op.
    /// </summary>
    /// <param name="op">Target operator.</param>
    /// <param name="context">Context.</param>
    /// <param name="iLmode">if is print is il or script.</param>
    /// <returns>Result.</returns>
    string PrintOp(Op op, IIRPrinterContext context, bool iLmode);

    /// <summary>
    /// if expr is callable will write to {dumpPath}/{prefix}_{callable.name}.{ext}`
    /// else write to {dumpPath}/{prefix}_{expr.Type.name}.il`.
    /// </summary>
    void DumpIR(Expr expr, string prefix, string dumpPath, bool display_callable);

    /// <summary>
    /// if expr is callable will write to {dumpPath}/{prefix}_{callable.name}.dot`.
    /// <remarks>
    /// not support prim func/prim func wrapper.
    /// </remarks>
    /// </summary>
    void DumpDotIR(Expr expr, string prefix, string dumpPath, bool display_callable);

    /// <summary>
    /// dump the expr as csharp code.
    /// </summary>
    /// <param name="expr">expression.</param>
    /// <param name="prefix">file prefix.</param>
    /// <param name="dumpDir">file dump ir.</param>
    /// <param name="randConst">false for save const into bin.</param>
    public void DumpCSharpIR(Expr expr, string prefix, string dumpDir, bool randConst);

    /// <summary>
    /// dump the expr as csharp code.
    /// </summary>
    /// <param name="expr">expression.</param>
    /// <param name="prefix">file prefix.</param>
    /// <param name="dumpDir">file dump ir.</param>
    public void DumpPatternIR(Expr expr, string prefix, string dumpDir);

    /// <summary>
    /// print ir type.
    /// </summary>
    string Print(IRType type);

    /// <summary>
    /// print ir type.
    /// </summary>
    /// <param name="expr"> the expression. </param>
    /// <param name="useScript">Print script format.</param>
    /// <returns>the string.</returns>
    string Print(Expr expr, bool useScript);

    /// <summary>
    /// Evaluate the expression tree.
    /// </summary>
    /// <param name="expr">Expression.</param>
    /// <param name="varsValues">Optional vars' values.</param>
    /// <param name="evaluator_cache"> Optional evaluator cache. </param>
    /// <returns>Evaluate result.</returns>
    IValue Evaluate(Expr expr, IReadOnlyDictionary<Var, IValue>? varsValues = null, Dictionary<Type, IEvaluator>? evaluator_cache = null);

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
    /// <returns>Evaluate result.</returns>
    Cost EvaluateCost(Expr expr);

    /// <summary>
    /// Evaluate metric of the expression tree.
    /// </summary>
    /// <param name="expr">Expression.</param>
    /// <returns>Evaluate result.</returns>
    Dictionary<Expr, Metric> EvaluateMetric(Expr expr);

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

    Expr EvaluateShapeExpr(Expr expr, ShapeExprCache? cache);

    Expr EvaluateOpShapeExpr(Op expr, IShapeEvaluateContext context);

    /// <summary>
    /// Match expression.
    /// </summary>
    /// <param name="expr">Expression to match.</param>
    /// <param name="pattern">Match pattern.</param>
    /// <param name="options">Match options.</param>
    /// <param name="result">Match result.</param>
    /// <returns>Match success.</returns>
    bool TryMatch(Expr expr, IPattern pattern, MatchOptions options, [MaybeNullWhen(false)] out IMatchResult result);

    /// <summary>
    /// Match expression as root.
    /// </summary>
    /// <param name="expr">Expression to match.</param>
    /// <param name="pattern">Match pattern.</param>
    /// <param name="options">Match options.</param>
    /// <param name="result">Match result.</param>
    /// <returns>Match success.</returns>
    bool TryMatchRoot(Expr expr, IPattern pattern, MatchOptions options, [MaybeNullWhen(false)] out IMatchResult result);

    /// <summary>
    /// Rewrite expression.
    /// </summary>
    /// <param name="expr">Expression.</param>
    /// <param name="rules">Rewrite rules.</param>
    /// <param name="options">Options.</param>
    /// <returns>Rewrited expression.</returns>
    Expr Rewrite(Expr expr, IEnumerable<IRewriteRule> rules, RunPassContext options);

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
    public bool TryEMatchRoot(Expr expr, IPattern pattern, [MaybeNullWhen(false)] out IReadOnlyList<IMatchResult> results);

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
    /// <returns>Rewrited expression.</returns>
    Expr ERewrite(Expr expr, IEnumerable<IRewriteRule> rules, RunPassContext options);

    /// <summary>
    /// Using EGraph rewrite expression.
    /// </summary>
    /// <param name="expr">Expression.</param>
    /// <param name="rules">Rewrite rules.</param>
    /// <param name="options">Options.</param>
    /// <returns>Rewrited expression.</returns>
    IEGraph ERewrite(IEGraph expr, IEnumerable<IRewriteRule> rules, RunPassContext options);
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

    /// <summary>
    /// Inference type of the expression tree.
    /// </summary>
    /// <param name="expr">Expression.</param>
    /// <returns>Is fully inferenced.</returns>
    public static bool InferenceType(this Expr expr)
    {
        return Provider.InferenceType(expr);
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
    public static IValue Evaluate(this Expr expr, IReadOnlyDictionary<Var, IValue>? varsValues = null, Dictionary<Type, IEvaluator>? evaluator_cache = null)
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
    /// <returns>Evaluate result.</returns>
    public static Cost EvaluateCost(Expr expr)
    {
        return Provider.EvaluateCost(expr);
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
    public static Dictionary<Expr, Metric> EvaluateMetric(Expr expr) => Provider.EvaluateMetric(expr);

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

    public static Expr EvaluateShapeExpr(this Expr expr, ShapeExprCache? cache = null)
    {
        return Provider.EvaluateShapeExpr(expr, cache);
    }

    public static Expr EvaluateOpShapeExpr(Op op, IShapeEvaluateContext context)
    {
        return Provider.EvaluateOpShapeExpr(op, context);
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
    public static bool TryMatchRoot(Expr expr, IPattern pattern, [MaybeNullWhen(false)] out IMatchResult result)
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
    public static bool TryMatchRoot(Expr expr, IPattern pattern, MatchOptions options, [MaybeNullWhen(false)] out IMatchResult result)
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
    public static Expr Rewrite(Expr expr, IEnumerable<IRewriteRule> rules, RunPassContext options)
    {
        return Provider.Rewrite(expr, rules, options);
    }

    /// <summary>
    /// Using EGraph rewrite expression.
    /// </summary>
    /// <param name="expr">Expression.</param>
    /// <param name="rules">Rewrite rules.</param>
    /// <param name="options">Options.</param>
    /// <returns>Rewrited expression.</returns>
    public static Expr ERewrite(Expr expr, IEnumerable<IRewriteRule> rules, RunPassContext options)
    {
        return Provider.ERewrite(expr, rules, options);
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
    /// <param name="iLmode">if is print is il or script.</param>
    /// <returns>Result.</returns>
    public static string PrintOp(Op op, IIRPrinterContext context, bool iLmode) => Provider.PrintOp(op, context, iLmode);

    public static void DumpIR(Expr expr, string prefix, string dumpPath, bool display_callable = true) =>
      Provider.DumpIR(expr, prefix, dumpPath, display_callable);

    /// <summary>
    /// if expr is callable will write to {dumpPath}/{prefix}_{callable.name}.dot`.
    /// <remarks>
    /// not support prim func/prim func wrapper.
    /// </remarks>
    /// </summary>
    public static void DumpDotIR(Expr expr, string prefix, string dumpPath, bool display_callable = true) =>
      Provider.DumpDotIR(expr, prefix, dumpPath, display_callable);

    /// <summary>
    /// dump the expr as csharp code.
    /// </summary>
    /// <param name="expr">expression.</param>
    /// <param name="prefix">file prefix.</param>
    /// <param name="dumpDir">file dump ir.</param>
    /// <param name="randConst">randConst = false will save the const into bin.</param>
    public static void DumpCSharpIR(Expr expr, string prefix, string dumpDir, bool randConst = true) =>
      Provider.DumpCSharpIR(expr, prefix, dumpDir, randConst);

    /// <summary>
    /// dump the expr as csharp code.
    /// </summary>
    /// <param name="expr">expression.</param>
    /// <param name="prefix">file prefix.</param>
    /// <param name="dumpDir">file dump ir.</param>
    public static void DumpPatternIR(Expr expr, string prefix, string dumpDir) =>
      Provider.DumpPatternIR(expr, prefix, dumpDir);

    public static string Print(IRType type) => Provider.Print(type);

    public static string Print(Expr expr, bool useScript = false) => Provider.Print(expr, useScript);

    /// <summary>
    /// Get target.
    /// </summary>
    /// <param name="name">Target name.</param>
    /// <returns>Target.</returns>
    public static ITarget GetTarget(string name) => Provider.GetTarget(name);

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
    private readonly IIRPrinterProvider _irprinterProvider;
    private readonly ICostEvaluateProvider _costEvaluateProvider;
    private readonly IMetricEvaluateProvider _metricEvaluateProvider;
    private readonly IMatchProvider _matchProvider;
    private readonly IRewriteProvider _rewriteProvider;
    private readonly IEGraphMatchProvider _eGraphMatchProvider;
    private readonly IEGraphRewriteProvider _eGraphrewriteProvider;
    private readonly ITargetProvider _targetProvider;
    private readonly IShapeEvaluateProvider _shapeEvaluateProvider;

    public CompilerServicesProvider(
        IEvaluateProvider evaluateProvider,
        ITypeInferenceProvider typeInferenceProvider,
        IIRPrinterProvider irprinterProvider,
        ICostEvaluateProvider costEvaluateProvider,
        IMetricEvaluateProvider metricEvaluateProvider,
        IDataTypeServiceProvider dataTypeServiceProvider,
        IMatchProvider matchProvider,
        IRewriteProvider rewriteProvider,
        IEGraphMatchProvider eGraphMatchProvider,
        IEGraphRewriteProvider eGraphrewriteProvider,
        ITargetProvider targetProvider,
        IShapeEvaluateProvider shapeEvaluateProvider)
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
        _eGraphMatchProvider = eGraphMatchProvider;
        _eGraphrewriteProvider = eGraphrewriteProvider;
        _targetProvider = targetProvider;
        _shapeEvaluateProvider = shapeEvaluateProvider;
    }

    public IDataTypeServiceProvider DataTypeService { get; }

    /// <inheritdoc/>
    public IValue Evaluate(Expr expr, IReadOnlyDictionary<Var, IValue>? varsValues = null, Dictionary<Type, IEvaluator>? evaluator_cache = null)
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
    public bool InferenceType(Expr expr)
    {
        return _typeInferenceProvider.InferenceType(expr);
    }

    /// <inheritdoc/>
    public string PrintOp(Op op, IIRPrinterContext context, bool iLmode)
    {
        return _irprinterProvider.PrintOp(op, context, iLmode);
    }

    /// <inheritdoc/>
    public void DumpIR(Expr expr, string prefix, string dumpPath, bool display_callable) =>
      _irprinterProvider.DumpIR(expr, prefix, dumpPath, display_callable);

    /// <inheritdoc/>
    public void DumpDotIR(Expr expr, string prefix, string dumpPath, bool display_callable) =>
    _irprinterProvider.DumpDotIR(expr, prefix, dumpPath, display_callable);

    /// <inheritdoc/>
    public void DumpCSharpIR(Expr expr, string prefix, string dumpDir, bool randConst) =>
    _irprinterProvider.DumpCSharpIR(expr, prefix, dumpDir, randConst);

    /// <inheritdoc/>
    public void DumpPatternIR(Expr expr, string prefix, string dumpDir) =>
    _irprinterProvider.DumpPatternIR(expr, prefix, dumpDir);

    /// <inheritdoc/>
    public string Print(IRType type) => _irprinterProvider.Print(type);

    /// <inheritdoc/>
    public string Print(Expr expr, bool useScript) => _irprinterProvider.Print(expr, useScript);

    /// <inheritdoc/>
    public bool TryMatch(Expr expr, IPattern pattern, MatchOptions options, [MaybeNullWhen(false)] out IMatchResult result)
    {
        return _matchProvider.TryMatch(expr, pattern, options, out result);
    }

    /// <inheritdoc/>
    public bool TryMatchRoot(Expr expr, IPattern pattern, MatchOptions options, [MaybeNullWhen(false)] out IMatchResult result)
    {
        return _matchProvider.TryMatchRoot(expr, pattern, options, out result);
    }

    /// <inheritdoc/>
    public Expr Rewrite(Expr expr, IEnumerable<IRewriteRule> rules, RunPassContext options)
    {
        return _rewriteProvider.Rewrite(expr, rules, options);
    }

    /// <inheritdoc/>
    public Cost EvaluateCost(Expr expr)
    {
        return _costEvaluateProvider.EvaluateCost(expr);
    }

    /// <inheritdoc/>
    public Cost EvaluateOpCost(Op op, ICostEvaluateContext context)
    {
        return _costEvaluateProvider.EvaluateOpCost(op, context);
    }

    /// <inheritdoc/>
    public Dictionary<Expr, Metric> EvaluateMetric(Expr expr) => _metricEvaluateProvider.EvaluateMetric(expr);

    /// <inheritdoc/>
    public Metric EvaluateOpMetric(Op op, IMetricEvaluateContext context) => _metricEvaluateProvider.EvaluateOpMetric(op, context);

    /// <inheritdoc/>
    public Expr EvaluateShapeExpr(Expr expr, ShapeExprCache? cache = null)
    {
        return _shapeEvaluateProvider.EvaluateShapeExpr(expr, cache ?? ShapeExprCache.Default);
    }

    /// <inheritdoc/>
    public Expr EvaluateOpShapeExpr(Op op, IShapeEvaluateContext context)
    {
        return _shapeEvaluateProvider.EvaluateOpShapeExpr(op, context);
    }

    public bool TryMatchRoot(IEnumerable<ENode> enodes, IPattern pattern, [MaybeNullWhen(false)] out IReadOnlyList<IMatchResult> results)
    {
        return _eGraphMatchProvider.TryMatchRoot(enodes, pattern, out results);
    }

    public bool TryEMatchRoot(Expr expr, IPattern pattern, [MaybeNullWhen(false)] out IReadOnlyList<IMatchResult> results)
    {
        return _eGraphMatchProvider.TryEMatchRoot(expr, pattern, out results);
    }

    public ITarget GetTarget(string name)
    {
        return _targetProvider.GetTarget(name);
    }

    public Expr ERewrite(Expr expr, IEnumerable<IRewriteRule> rules, RunPassContext options)
    {
        return _eGraphrewriteProvider.ERewrite(expr, rules, options);
    }

    public IEGraph ERewrite(IEGraph graph, IEnumerable<IRewriteRule> rules, RunPassContext options)
    {
        return _eGraphrewriteProvider.ERewrite(graph, rules, options);
    }
}
