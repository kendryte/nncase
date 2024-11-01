// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Nncase.CostModel;
using Nncase.IR;
using Nncase.PatternMatch;

namespace Nncase.IR.CustomCPU;

[PatternFunctionalGenerator]
public sealed partial class Unary : Op
{
    public static readonly ParameterInfo Input = new(typeof(Unary), 0, "input", ParameterKind.Input);

    public UnaryOp UnaryOp { get; }

    public IRArray<SBP> SBPs { get; }

    public Cost Cost { get; }

    public string CSourcePath { get; }

    /// <inheritdoc/>
    public override string DisplayProperty()
    {
        return $"CustomUnaryOp.{UnaryOp}";
    }
}
