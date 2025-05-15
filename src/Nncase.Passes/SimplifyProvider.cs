// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections;
using System.Collections.Generic;
using System.CommandLine;
using System.CommandLine.Invocation;
using System.Diagnostics.CodeAnalysis;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Threading.Tasks;
using Microsoft.Extensions.Configuration;
using Nncase.CodeGen;
using Nncase.Diagnostics;
using Nncase.IR;
using Nncase.Passes.Mutators;
using Nncase.Passes.Rules.Neutral;
using Nncase.Passes.Rules.ShapeExpr;
using Nncase.Passes.Transforms;
using Nncase.Quantization;
using Nncase.Targets;

namespace Nncase.Passes;

internal sealed class SimplifyTarget : ITarget
{
    public string Name => "Simplify";

    public IReadOnlyList<IModuleCompiler> ModuleCompilers => throw new NotImplementedException();

    public Task AdaRoundWeights(ICalibrationDatasetProvider calibrationDataset, List<ENode> rangeOfs, List<ENode> childrenOfRangeOfs, QuantizeOptions quantizeOptions) => throw new NotImplementedException();

    public Task<Dictionary<ENode, List<Tuple<List<DataType>, List<List<QuantParam>>, float>>>> BindQuantMethodCosine(ICalibrationDatasetProvider calibrationDataset, List<ENode> rangeOfs, List<ENode> childrenOfRangeOfs, QuantizeOptions quantizeOptions) => throw new NotImplementedException();

    public IModuleBuilder CreateModuleBuilder(string moduleKind, CompileOptions options) => throw new NotImplementedException();

    public IModuleCompiler GetModuleCompiler(string moduleKind) => throw new NotImplementedException();

    public void ParseTargetDependentOptions(IConfigurationSection configure) => throw new NotImplementedException();

    public void RegisterAffineSelectionPass(IPassManager passManager, CompileOptions options) => throw new NotImplementedException();

    public void RegisterAutoPackingRules(IRulesAddable pass, CompileOptions options) => throw new NotImplementedException();

    public void RegisterPostAutoPackingPass(IPassManager passManager, CompileOptions options) => throw new NotImplementedException();

    public (Command Command, Func<InvocationContext, Command, ITargetOptions> Parser) RegisterCommandAndParser() => throw new NotImplementedException();

    public void RegisterQuantizePass(IPassManager passManager, CompileOptions options) => throw new NotImplementedException();

    public void RegisterPostQuantizePass(IPassManager passManager, CompileOptions options) => throw new NotImplementedException();

    public void RegisterTargetDependentAfterQuantPass(IPassManager passManager, CompileOptions options) => throw new NotImplementedException();

    public void RegisterTargetDependentBeforeCodeGen(IPassManager passManager, CompileOptions options) => throw new NotImplementedException();

    public void RegisterTargetDependentPass(IPassManager passManager, CompileOptions options) => throw new NotImplementedException();

    public void RegisterTargetInDependentPass(IPassManager passManager, CompileOptions options) => throw new NotImplementedException();

    public void RegisterTIRSelectionPass(IPassManager passManager, CompileOptions optionsÍ) => throw new NotImplementedException();
}

internal sealed class SimplifyProvider : ISimplifyProvider
{
    private readonly CompileSession _compileSession;
    private readonly IRewriteRule[] _rules;

    public SimplifyProvider()
    {
        _compileSession = CompileSession.Create(new SimplifyTarget(), new CompileOptions());
        using var compileScope = new CompileSessionScope(_compileSession);
        _rules = [
            new Rules.Neutral.FoldConstCall(),
            new SliceToGetItem(),
            new GatherToGetItem(),
            new FoldGetItemShapeOf(),
            new FoldGetItemConcat(),
            new FoldGetItemReshape(),
            new FoldSplitShapeOf(),
        ];
    }

    public Expr SimplifyForDimension(Expr expr)
    {
#if true
        if (expr.CheckedType is DistributedType || expr is not (Const or Var))
        {
            var simplifiedExpr = expr;
            if (expr.CheckedType is DistributedType)
            {
                simplifiedExpr = new RemoveBoxingCloner().Clone(expr, default);
            }

            using var compileScope = new CompileSessionScope(CompileSessionScope.Current ?? _compileSession);
            using var dumpScope = new DumpScope(NullDumpper.Instance);
            simplifiedExpr = (Expr)CompilerServices.Rewrite(simplifiedExpr, _rules, new RunPassContext());
            return simplifiedExpr;
        }

        return expr;
#else
        return expr;
#endif
    }

    public bool TryGetMaxShape(Shape shape, [MaybeNullWhen(false)] out long[] maxShape)
    {
        if (shape.IsFixed)
        {
            maxShape = shape.ToValueArray();
            return true;
        }

        if (shape is RankedShape rankedShape)
        {
            maxShape = new long[rankedShape.Rank];
            if (!rankedShape.Metadata.Range.HasValue)
            {
                new InferRangeVisitor().Visit(rankedShape);
            }

            for (int i = 0; i < rankedShape.Rank; i++)
            {
                var max = rankedShape.Dimensions[i].Metadata.Range!.Value.Max;
                if (max >= int.MaxValue)
                {
                    maxShape = null;
                    return false;
                }

                maxShape[i] = (long)max;
            }

            return true;
        }

        maxShape = null;
        return false;
    }
}
