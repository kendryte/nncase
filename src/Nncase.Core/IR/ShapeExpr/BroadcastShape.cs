// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using static Nncase.IR.TypePatternUtility;

namespace Nncase.IR.ShapeExpr;

public class BroadcastShape : ShapeExprOp
{
    public static readonly ParameterInfo Inputs = new(typeof(BroadcastShape), 0, "inputs");
}
