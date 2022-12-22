// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

namespace Nncase;

/// <summary>
/// Reduce operator.
/// </summary>
public enum ReduceOp : byte
{
    /// <summary>
    /// Mean.
    /// </summary>
    Mean,

    /// <summary>
    /// Min.
    /// </summary>
    Min,

    /// <summary>
    /// Max.
    /// </summary>
    Max,

    /// <summary>
    /// Sum.
    /// </summary>
    Sum,

    /// <summary>
    /// Product.
    /// </summary>
    Prod,
}
