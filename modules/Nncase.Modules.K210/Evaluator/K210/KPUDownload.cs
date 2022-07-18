using System;
using Nncase.CostModel;
using Nncase.IR;
using Nncase.IR.K210;
using OrtKISharp;

namespace Nncase.Evaluator.K210;

/// <summary>
/// Evaluator for <see cref="KPUDownload"/>.
/// </summary>
internal sealed class KPUDownloadEvaluator : IEvaluator<KPUDownload>, ITypeInferencer<KPUDownload>, ICostEvaluator<KPUDownload>
{
    /// <inheritdoc/>
    public IValue Visit(IEvaluateContext context, KPUDownload target)
    {
        var input = context.GetArgumentValue(target, KPUDownload.Input);
        return input;
    }

    /// <inheritdoc/>
    public IRType Visit(ITypeInferenceContext context, KPUDownload target)
    {
        var input = context.CheckArgumentType<TensorType>(target, KPUDownload.Input);
        return Visit(target, input);
    }

    /// <inheritdoc/>
    public Cost Visit(ICostEvaluateContext context, KPUDownload target)
    {
        var inputType = context.GetArgumentType<TensorType>(target, KPUDownload.Input);
        var outputType = context.GetReturnType<TensorType>();

        return new()
        {
            [CostFactorNames.MemoryLoad] = CostUtility.GetMemoryAccess(inputType),
            [CostFactorNames.MemoryStore] = CostUtility.GetMemoryAccess(outputType),
        };
    }

    private IRType Visit(KPUDownload target, TensorType input)
    {
        return input;
    }
}