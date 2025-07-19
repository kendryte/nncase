// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Nncase.CostModel;
using Nncase.IR;
using Nncase.PatternMatch;

namespace Nncase.IR.CustomNTT;

[PatternFunctionalGenerator]
public sealed partial class Unary : Op
{
    public static readonly ParameterInfo Input = new(typeof(Unary), 0, "input", ParameterKind.Input);

    public UnaryOp UnaryOp { get; }

    public IRArray<SBP> InSBPs { get; }

    public IRArray<SBP> OutSBPs { get; }

    public Cost Cost { get; }

    public string CSourcePath { get; }

    public string FuncName { get; }

    /// <inheritdoc/>
    public override string DisplayProperty()
    {
        return $"CustomUnaryOp.{UnaryOp}";
    }
}
