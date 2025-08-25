// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Nncase.IR.Shapes;

namespace Nncase.IR.F;

/// <summary>
/// Shapes functional helper.
/// </summary>
public static class Shapes
{
    public static Call AsTensor(Dimension input) => new Call(new AsTensor(), input);

    public static Call AsTensor(Shape input) => new Call(new AsTensor(), input);

    /// <summary>
    /// Call compare.
    /// </summary>
    /// <param name="compareOp">Compare operator.</param>
    /// <param name="lhs">Left operand.</param>
    /// <param name="rhs">Right operand.</param>
    /// <param name="maskVectorStyle">Mask vector style.</param>
    /// <returns>Result expression.</returns>
    public static Call Compare(CompareOp compareOp, Dimension lhs, Dimension rhs, MaskVectorStyle maskVectorStyle = MaskVectorStyle.Unknown) => new Call(new IR.Math.Compare(compareOp, maskVectorStyle), lhs, rhs);

    /// <summary>
    /// Call equal.
    /// </summary>
    public static Call Equal(Dimension lhs, Dimension rhs) => Compare(CompareOp.Equal, lhs, rhs);

    /// <summary>
    /// call not equal.
    /// </summary>
    public static Call NotEqual(Dimension lhs, Dimension rhs) => Compare(CompareOp.NotEqual, lhs, rhs);

    /// <summary>
    /// call less than.
    /// </summary>
    public static Call LessThan(Dimension lhs, Dimension rhs) => Compare(CompareOp.LowerThan, lhs, rhs);

    /// <summary>
    /// call less equal.
    /// </summary>
    public static Call LessEqual(Dimension lhs, Dimension rhs) => Compare(CompareOp.LowerOrEqual, lhs, rhs);

    /// <summary>
    /// call greater equal.
    /// </summary>
    public static Call GreaterEqual(Dimension lhs, Dimension rhs) => Compare(CompareOp.GreaterOrEqual, lhs, rhs);

    /// <summary>
    /// call greater than.
    /// </summary>
    public static Call GreaterThan(Dimension lhs, Dimension rhs) => Compare(CompareOp.GreaterThan, lhs, rhs);
}
