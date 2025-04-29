// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

namespace Nncase.IR.Shapes;

public static class ShapeBuilder
{
    public static RankedShape Create(ReadOnlySpan<int> dimensions) => new RankedShape(dimensions);

    public static RankedShape Create(ReadOnlySpan<long> dimensions) => new RankedShape(dimensions);

    public static RankedShape Create(ReadOnlySpan<Dimension> dimensions) => new RankedShape(dimensions);
}
