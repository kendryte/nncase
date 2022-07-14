// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using NetFabric.Hyperlinq;
using Nncase.IR;
using Nncase.IR.NN;
using Nncase.IR.Tensors;
using OrtKISharp;
using static OrtKISharp.TensorHelper;

namespace Nncase.Evaluator.NN;

/// <summary>
/// Evaluator for <see cref="CustomCall"/>.
/// </summary>
public class CustomCallEvaluator : IEvaluator<CustomCall>, ITypeInferencer<CustomCall>
{
    /// <inheritdoc/>
    public IRType Visit(ITypeInferenceContext context, CustomCall target)
    {
        return CompilerServices.InferenceOp(target.CustomOp, context);
    }

    /// <inheritdoc/>
    public IValue Visit(IEvaluateContext context, CustomCall target)
    {
        return CompilerServices.EvaluateOp(target.CustomOp, context);
    }
}
