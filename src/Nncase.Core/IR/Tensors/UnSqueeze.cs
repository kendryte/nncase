// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Nncase.PatternMatch;
using static Nncase.IR.TypePatternUtility;

namespace Nncase.IR.Tensors;

/// <summary>
/// Unsqueeze expression.
/// <see cref="http://www.xavierdupre.fr/app/mlprodict/helpsphinx/onnxops/onnx__Unsqueeze.html#unsqueeze-13"/>
/// NOTE Dim will apply by sequence. 
/// </summary>
[PatternFunctionalGenerator]
public sealed record Unsqueeze() : Op
{
    /// <summary>
    /// Gets input.
    /// </summary>
    public static readonly ParameterInfo Input = new(typeof(Unsqueeze), 0, "input");

    /// <summary>
    /// Gets dimension.
    /// </summary>
    public static ParameterInfo Dim = new(typeof(Unsqueeze), 1, "dim", HasRank(1) & IsIntegral());
}
