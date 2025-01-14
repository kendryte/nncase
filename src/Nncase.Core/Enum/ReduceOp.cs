// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System.Text.Json.Serialization;

namespace Nncase;

/// <summary>
/// Reduce operator.
/// </summary>
[JsonConverter(typeof(JsonStringEnumConverter))]
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
