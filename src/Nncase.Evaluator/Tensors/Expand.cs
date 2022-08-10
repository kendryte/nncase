// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.IO;
using System.Linq;
using NetFabric.Hyperlinq;
using Nncase.IR;
using Nncase.IR.Math;
using Nncase.IR.Tensors;
using OrtKISharp;
using Shape = Nncase.IR.Shape;

namespace Nncase.Evaluator.Tensors;

/// <summary>
/// Evaluator for <see cref="Expand"/>.
/// </summary>
[TypeInferGenerator]
public sealed partial class ExpandEvaluator : IEvaluator<Expand>, ITypeInferencer<Expand>
{
    /// <inheritdoc/>
    public IValue Visit(IEvaluateContext context, Expand expand)
    {
        var input = context.GetOrtArgumentValue(expand, Expand.Input);
        var shape = context.GetInt64OrtTensorArgumentValue(expand, Expand.Shape);
        return OrtKI.Expand(input, shape).ToValue();
    }

    private IRType Visit(ITypeInferenceContext context, Expand target, TensorType Input, TensorType Shape)
    {
        var shape_expr = context.GetArgument(target, Expand.Shape);
        if (shape_expr is TensorConst constShape)
            return Input with { Shape = new Shape(constShape.Value.Cast<int>()) };
        else
            return Input with { Shape = TypeInference.ReshapeTo(Shape) };
    }
}
