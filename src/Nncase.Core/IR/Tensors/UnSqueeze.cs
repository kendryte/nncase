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
/// NOTE Dim will apply by sequence.
/// </summary>
/// <remarks>"http://www.xavierdupre.fr/app/mlprodict/helpsphinx/onnxops/onnx__Unsqueeze.html#unsqueeze-13".</remarks>
[PatternFunctionalGenerator]
public sealed partial class Unsqueeze : Op
{
    /// <summary>
    /// Gets input.
    /// </summary>
    public static readonly ParameterInfo Input = new(typeof(Unsqueeze), 0, "input");

    /// <summary>
    /// Gets dimension.
    /// </summary>
    public static readonly ParameterInfo Dim = new(typeof(Unsqueeze), 1, "dim", HasRank(1) & IsIntegral());
}
