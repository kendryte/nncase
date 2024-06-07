// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Microsoft.Extensions.DependencyInjection;
using Nncase.Diagnostics;
using Nncase.IR;
using Nncase.TIR;

namespace Nncase.Passes;

internal sealed class PassManager : IPassManager
{
    private readonly CompileSession _compileSession;
    private readonly IDumpper _dummper;
    private readonly List<PassGroup> _passesGroups;

    private int _currentPassIndex;
    private bool _isLastAddedEGraphPass;
    private FunctionPassGroup _lastFunctionPassGroup;
    private bool _freezed;

    /// <summary>
    /// Initializes a new instance of the <see cref="PassManager"/> class.
    /// </summary>
    /// <param name="name">Pass manager name.</param>
    /// <param name="compileSession">Compile session.</param>
    public PassManager(string name, CompileSession compileSession)
    {
        Name = name;
        _compileSession = compileSession;
        _dummper = DumpScope.GetCurrent(compileSession).CreateSubDummper(name, null);
        _lastFunctionPassGroup = new(_compileSession, _currentPassIndex);
        _passesGroups = new() { _lastFunctionPassGroup };
    }

    /// <summary>
    /// Gets name.
    /// </summary>
    public string Name { get; }

    /// <inheritdoc/>
    public AddPassResult<T> Add<T>(params object[] parameters)
        where T : class, IPass
    {
        EnsureNotFreezed();
        var pass = ActivatePass<T>(parameters);
        switch (pass)
        {
            case EGraphPass:
                if (!_isLastAddedEGraphPass)
                {
                    AddPass(ActivatePass<EGraphConstructPass>());
                }

                AddPass(pass);
                _isLastAddedEGraphPass = true;
                break;
            case FunctionPass or PrimFuncPass:
                if (_isLastAddedEGraphPass)
                {
                    AddPass(ActivatePass<EGraphExtractPass>());
                }

                AddPass(pass);
                _isLastAddedEGraphPass = false;
                break;
            case ModulePass modulePass:
                if (_isLastAddedEGraphPass)
                {
                    AddPass(ActivatePass<EGraphExtractPass>());
                }

                AddModulePass(modulePass);
                _isLastAddedEGraphPass = false;
                break;
            case EGraphExtractPass:
                if (!_isLastAddedEGraphPass)
                {
                    throw new InvalidOperationException("Can't Add EGraphExtractPass without EGraphPass!");
                }

                AddPass(pass);
                _isLastAddedEGraphPass = false;
                break;
            default:
                throw new NotSupportedException($"Unsupported pass type: {pass.GetType().AssemblyQualifiedName}");
        }

        return new(this, _compileSession, pass);
    }

    /// <inheritdoc/>
    public AddPassResult<T> AddWithName<T>(string name, params object[] parameters)
        where T : class, IPass
    {
        var result = Add<T>(parameters);
        result.Configure(p => p.Name = name);
        return result;
    }

    /// <inheritdoc/>
    public async Task<IRModule> RunAsync(IRModule module)
    {
        Freeze();

        using var dumpScope = new DumpScope(_dummper);
        foreach (var group in _passesGroups)
        {
            module = await group.RunAsync(module);
        }

        return module;
    }

    private T ActivatePass<T>(params object[] parameters)
        where T : class, IPass
    {
        using var scope = new CompileSessionScope(_compileSession);
        using var dumpScope = new DumpScope(_dummper);
        var pass = ActivatorUtilities.CreateInstance<T>(_compileSession, parameters);
        return pass;
    }

    private void AddPass(IPass pass)
    {
        _lastFunctionPassGroup.Passes.Add(pass);
        _currentPassIndex++;
    }

    private void AddModulePass(ModulePass modulePass)
    {
        _passesGroups.Add(new ModulePassGroup(modulePass, _currentPassIndex++));
        _lastFunctionPassGroup = new(_compileSession, _currentPassIndex);
        _passesGroups.Add(_lastFunctionPassGroup);
    }

    private void EnsureNotFreezed()
    {
        if (_freezed)
        {
            throw new InvalidOperationException("PassManager is freezed.");
        }
    }

    private void Freeze()
    {
        if (!_freezed)
        {
            if (_isLastAddedEGraphPass)
            {
                AddPass(ActivatePass<EGraphExtractPass>());
                _isLastAddedEGraphPass = false;
            }

            _freezed = true;
        }
    }

    private abstract class PassGroup
    {
        public PassGroup(int startPassIndex)
        {
            StartPassIndex = startPassIndex;
        }

        protected int StartPassIndex { get; }

        public abstract Task<IRModule> RunAsync(IRModule module);
    }

    private class FunctionPassGroup : PassGroup
    {
        private readonly CompileSession _compileSession;

        public FunctionPassGroup(CompileSession compileSession, int startPassIndex)
            : base(startPassIndex)
        {
            _compileSession = compileSession;
        }

        public List<IPass> Passes { get; } = new();

        public override async Task<IRModule> RunAsync(IRModule module)
        {
            bool replaced = false;
            for (int i = 0; i < module.Functions.Count; i++)
            {
                var pre = module.Functions[i];
                var runner = new Runner(_compileSession, pre, Passes, StartPassIndex);
                var post = await runner.RunAsync();

                if (!object.ReferenceEquals(pre, post))
                {
                    module.Replace(i, post);
                    replaced = true;
                }
            }

            if (replaced && DumpScope.Current.IsEnabled(DumpFlags.PassIR))
            {
                DumpScope.Current.DumpModule(module, $"FunctionCallUpdate_{StartPassIndex}/After");
            }

            return module;
        }

        private class Runner
        {
            private readonly CompileSession _compileSession;
            private readonly IAnalyzerManager _analyzerManager;
            private readonly IReadOnlyList<IPass> _passes;
            private readonly int _startPassIndex;

            private BaseFunction _function;
            private IEGraph? _eGraph;

            public Runner(CompileSession compileSession, BaseFunction function, IReadOnlyList<IPass> passes, int startPassIndex)
            {
                _analyzerManager = compileSession.GetRequiredService<IAnalyzerManager>();
                _compileSession = compileSession;
                _function = function;
                _passes = passes;
                _startPassIndex = startPassIndex;
            }

            public async Task<BaseFunction> RunAsync()
            {
                for (int i = 0; i < _passes.Count; i++)
                {
                    var pass = _passes[i];
                    var context = CreateRunPassContext(pass, _startPassIndex + i);
                    switch (pass)
                    {
                        case FunctionPass p:
                            _function = await p.RunAsync(_function, context);
                            break;
                        case PrimFuncPass p:
                            if (_function is PrimFunction pf)
                            {
                                _function = await p.RunAsync(pf, context);
                            }

                            break;
                        case EGraphPass p:
                            _eGraph = await p.RunAsync(_eGraph!, context);
                            break;
                        case EGraphConstructPass p:
                            _eGraph = await p.RunAsync(_function, context);
                            break;
                        case EGraphExtractPass p:
                            _function = await p.RunAsync(_eGraph!, context);
                            _eGraph = null;
                            break;
                        default:
                            throw new NotSupportedException($"Unsupported pass type: {pass.GetType().AssemblyQualifiedName}");
                    }
                }

                return _function;
            }

            private RunPassContext CreateRunPassContext(IPass pass, int index)
            {
                var context = pass switch
                {
                    FunctionPass or PrimFuncPass or EGraphConstructPass => new RunPassContextWithAnalysis(_analyzerManager, _function, pass),
                    EGraphPass or EGraphExtractPass => new RunPassContextWithAnalysis(_analyzerManager, Either<BaseFunction, IEGraph>.From(_eGraph!), pass),
                    _ => throw new NotSupportedException($"Unsupported pass type: {pass.GetType().AssemblyQualifiedName}"),
                };
                context.Index = index;
                return context;
            }

            private sealed record RunPassContextWithAnalysis : RunPassContext
            {
                private readonly IReadOnlyList<IAnalyzer> _analyzers;

                public RunPassContextWithAnalysis(IAnalyzerManager analyzerManager, Either<BaseFunction, IEGraph> functionOrEGraph, IPass pass)
                {
                    var populater = new AnalyzerPopulater(analyzerManager, functionOrEGraph);
                    populater.Populate(pass.AnalysisTypes);
                    _analyzers = populater.Analyzers;
                    AnalysisResults = populater.AnalysisResults;
                    RewriteOnce = _analyzers.Count != 0;
                    Driver = pass;
                }

                private struct AnalyzerPopulater
                {
                    private readonly IAnalyzerManager _analyzerManager;
                    private readonly Either<BaseFunction, IEGraph> _functionOrEGraph;

                    public AnalyzerPopulater(IAnalyzerManager analyzerManager, Either<BaseFunction, IEGraph> functionOrEGraph)
                    {
                        _analyzerManager = analyzerManager;
                        _functionOrEGraph = functionOrEGraph;
                    }

                    public Dictionary<Type, IAnalysisResult> AnalysisResults { get; } = new();

                    public List<IAnalyzer> Analyzers { get; } = new();

                    public void Populate(IReadOnlyCollection<Type> analysisResultTypes)
                    {
                        foreach (var type in analysisResultTypes)
                        {
                            if (!AnalysisResults.ContainsKey(type))
                            {
                                var factory = _analyzerManager.GetFactory(type);
                                var analyzer = factory.Activate(_functionOrEGraph);
                                Populate(analyzer.RequiredAnalysisResultTypes);
                                Analyzers.Add(analyzer);
                                AnalysisResults.Add(type, analyzer.Result);
                            }
                        }
                    }
                }
            }
        }
    }

    private class ModulePassGroup : PassGroup
    {
        private readonly ModulePass _modulePass;

        public ModulePassGroup(ModulePass modulePass, int startPassIndex)
            : base(startPassIndex)
        {
            _modulePass = modulePass;
        }

        public override async Task<IRModule> RunAsync(IRModule module)
        {
            var context = new RunPassContext { Index = StartPassIndex };
            return await _modulePass.RunAsync(module, context);
        }
    }
}

internal sealed class PassManagerFactory : IPassManagerFactory
{
    public IPassManager Create(string name, CompileSession compileSession)
        => new PassManager(name, compileSession);
}
