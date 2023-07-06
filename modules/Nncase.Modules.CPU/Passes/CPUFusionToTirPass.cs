using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Microsoft.Extensions.DependencyInjection;
using Nncase.Diagnostics;
using Nncase.IR;
using Nncase.Passes.Analysis;
using Nncase.Passes.Mutators;
using Nncase.Passes.Tile;
using Nncase.TIR;

namespace Nncase.Passes;

internal sealed class CPUFusionToTirPass : ModulePass
{
    private readonly TileOptions _tileOptions;
    private readonly Dictionary<Fusion, ulong> _fusionMacsMap;

    public CPUFusionToTirPass(TileOptions tileOptions)
    {
        _tileOptions = tileOptions;
        _fusionMacsMap = new(ReferenceEqualityComparer.Instance);
    }

    private IAnalyzerManager AnalyzerManager => CompileSession.GetRequiredService<IAnalyzerManager>();

    /// <inheritdoc/>
    protected override Task<IRModule> RunCoreAsync(IRModule module, RunPassContext options)
    {
        Dictionary<Fusion, BaseFunction> fusionConertedCache = new(ReferenceEqualityComparer.Instance);

        // convert the fusion as entry.
        for (int i = 0; i < module.Functions.Count; i++)
        {
            if (module.Functions[i] is Fusion { ModuleKind: "cpu" } fusion)
            {
                TIR.PrimFunction primFunction;
                var visitor = new MultiLayerFusionConverter(_tileOptions);
                primFunction = visitor.VisitToPrimFunc(fusion);

                CompilerServices.InferenceType(primFunction);
                fusionConertedCache[fusion] = primFunction;
                module.Replace(i, primFunction);
            }
        }

        // convert the stackvm function call k510 fusion
        for (int i = 0; i < module.Functions.Count; i++)
        {
            if (module.Functions[i] is Function { ModuleKind: "stackvm" } func)
            {
                var analysis = new Dictionary<Type, IAnalysisResult>
                {
                    [typeof(IExprUserAnalysisResult)] = AnalyzerManager.GetAnaylsis<IExprUserAnalysisResult>(func),
                };
                var rewriter = new DataFlowMergeRewriter();
                var fusionCheckCache = new Dictionary<Fusion, IFusionChecker>(ReferenceEqualityComparer.Instance);

                var post = (Function)rewriter.Rewrite(func, new Mutators.IMergeRewriteRule[] { new GNNESameInputFusionMergeRule(), }, (rule, option) => new CPUFusionGroupMutator<MultiFusionChecker>(fusionCheckCache, _tileOptions, rule, option), new() { AnalysisResults = analysis, MatchOptions = new Mutators.FusionGroupMutator.GroupedMatchOptions() });

                // if (DumpScope.Current.IsEnabled(DumpFlags.PassIR))
                // {
                //     DumpScope.Current.DumpDotIR(post, "MultiLayer");
                // }
                // post = (Function)rewriter.Rewrite(
                //    post,
                //    new Mutators.IMergeRewriteRule[] {
                //    new GNNESameInputFusionMergeRule(),
                //  },
                //    (rule, option) => new CPUFusionGroupMutator<TwoFusionChecker>(fusionCheckCache, _tileOptions, rule, option),
                //    new() { AnalysisResults = analysis, MatchOptions = new Mutators.FusionGroupMutator.GroupedMatchOptions() });

                // if (DumpScope.Current.IsEnabled(DumpFlags.PassIR))
                // {
                //     DumpScope.Current.DumpDotIR(post, "TwoLayer");
                // }
                // var post = func;
                var mutator = new CheckedConvertMutator(fusionConertedCache, _fusionMacsMap, fusionCheckCache, _tileOptions, options);
                var new_func = (Function)mutator.Rewrite(post);
                CompilerServices.InferenceType(new_func);
                if (mutator.IsMutated)
                {
                    module.Replace(i, new_func);
                }
            }
        }

        // add all prim func.
        foreach (var item in fusionConertedCache.Values)
        {
            if (item is PrimFunctionWrapper wrapper)
            {
                module.Add(wrapper);
                module.Add(wrapper.Target);
            }
        }

        return Task.FromResult(module);
    }

    protected override async Task OnPassEndAsync(IRModule post, RunPassContext context)
    {
        await base.OnPassEndAsync(post, context);
        if (DumpScope.Current.IsEnabled(DumpFlags.PassIR))
        {
            using var writer = new StreamWriter(DumpScope.Current.OpenFile("mac.csv"));
            foreach (var (fusion, mac) in _fusionMacsMap)
            {
                writer.WriteLine($"mac: {fusion.Name},{mac}");
            }
        }

        _fusionMacsMap.Clear();
    }
}
