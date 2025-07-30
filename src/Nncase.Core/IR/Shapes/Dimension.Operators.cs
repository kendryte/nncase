// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

namespace Nncase.IR;

public abstract partial class Dimension
{
    /// <summary>
    /// GreaterEqual.
    /// </summary>
    public static Call operator >=(Dimension lhs, Dimension rhs) => IR.F.Shapes.GreaterEqual(lhs, rhs);

    /// <summary>
    /// GreaterThan.
    /// </summary>
    public static Call operator >(Dimension lhs, Dimension rhs) => IR.F.Shapes.GreaterThan(lhs, rhs);

    /// <summary>
    /// LessEqual.
    /// </summary>
    public static Call operator <=(Dimension lhs, Dimension rhs) => IR.F.Shapes.LessEqual(lhs, rhs);

    /// <summary>
    /// LessThan.
    /// </summary>
    public static Call operator <(Dimension lhs, Dimension rhs) => IR.F.Shapes.LessThan(lhs, rhs);
}
