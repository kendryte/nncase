// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

namespace Nncase.TIR;

/// <summary>
/// For loop mode.
/// </summary>
public enum LoopMode
{
    /// <summary>
    /// default semantics -- serial execution.
    /// </summary>
    Serial,

    /// <summary>
    /// Parallel execution on CPU.
    /// </summary>
    Parallel,

    /// <summary>
    /// Vector SIMD loop.
    ///  The loop body will be vectorized.
    /// </summary>
    Vectorized,

    /// <summary>
    /// The loop body must be unrolled.
    /// </summary>
    Unrolled,

    /// <summary>
    /// The loop variable is bound to a thread in
    /// an environment. In the final stage of lowering,
    /// the loop is simply removed and the loop variable is
    /// mapped to the corresponding context thread.
    /// </summary>
    ThreadBinding,
}
