// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using Nncase.CostModel;
using Nncase.IR;
using Nncase.IR.CPU;
using Nncase.Utilities;
using OrtKISharp;

namespace Nncase.Evaluator.IR.CPU;

public sealed class PackedTransposeEvaluator : IEvaluator<PackedTranspose>, ITypeInferencer<PackedTranspose>, ICostEvaluator<PackedTranspose>
{
    public IValue Visit(IEvaluateContext context, PackedTranspose target)
    {
        var input = context.GetOrtArgumentValue(target, PackedTranspose.Input);
        var perm = context.GetArgumentValueAsArray<long>(target, PackedTranspose.Perm);

        var packedAxes = target.PackedAxes.Select(axis => perm.IndexOf(axis)).ToArray();
        var restAxis = LinqUtility.Range<long>(perm.Length, packedAxes.Length).ToArray();
        restAxis = packedAxes.Zip(restAxis).OrderBy(p => p.First).Select(p => p.Second).ToArray();

        perm = perm.Concat(restAxis).ToArray();

        var transposed = OrtKI.Transpose(input, perm);

        return Value.FromTensor(Tensor.FromBytes(context.CurrentCall.CheckedDataType, transposed.BytesBuffer.ToArray(), context.CurrentCall.CheckedShape.ToValueArray()));
    }

    public IRType Visit(ITypeInferenceContext context, PackedTranspose target)
    {
        var input = context.CheckArgumentType<IRType>(target, PackedTranspose.Input);
        var permExpr = context.GetArgument(target, PackedTranspose.Perm);

        return input switch
        {
            DistributedType d => Tensors.TransposeEvaluator.Visit(d, permExpr),
            TensorType t => Tensors.TransposeEvaluator.Visit(t, permExpr),
            AnyType => AnyType.Default,
            _ => new InvalidType(input.GetType().ToString()),
        };
    }

    /// <inheritdoc/>
    public Cost Visit(ICostEvaluateContext context, PackedTranspose target)
    {
        var inputType = context.GetArgumentType<IRType>(target, PackedTranspose.Input);
        var outputType = context.GetReturnType<IRType>();

        return new()
        {
            [CostFactorNames.MemoryLoad] = CostUtility.GetMemoryAccess(inputType),
            [CostFactorNames.MemoryStore] = CostUtility.GetMemoryAccess(outputType),
        };
    }
}
