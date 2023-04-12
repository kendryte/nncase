// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

namespace Nncase.TIR;

/// <summary>
/// Iteration mode.
/// </summary>
public enum IterationMode
{
    /// <summary>
    /// Data parallel iteration.
    ///  This normally corresponds to axis of Tensor.
    ///  Allow all IterVar manipulations.
    /// <remarks>
    /// This does not mean the loop have to be executed in parallel fashion.
    /// </remarks>
    /// </summary>
    DataParallel,

    /// <summary>
    /// The IterVar itself is a thread-index of a fixed thread launching group.  Note that this is already assumed to be parallelized.
    /// <remarks>
    /// Disallow: split/fuse/vectorize/parallel
    /// </remarks>
    /// </summary>
    ThreadIndex,

    /// <summary>
    /// Communicative reduction.
    ///  Cannot be directly parallelized.
    /// <remarks>
    /// Disallow: parallel/vectorize
    /// </remarks>
    /// </summary>
    CommReduce,

    /// <summary>
    /// Serial loops with loop carry dependency,
    ///  the iteration must execute in order.
    ///  Cannot be re-ordered.
    /// <remarks>
    /// Disallow: reorder/parallel/vectorize
    /// </remarks>
    /// </summary>
    Ordered,

    /// <summary>
    /// IterVar is opaque,
    /// May not corresponds to any generated loop
    /// Disallow all IterVar manipulations and compute_at.
    /// <remarks>
    /// This is usually used to implement composite op or external op, where the
    /// </remarks>
    /// </summary>
    Opaque,

    // The following are possible additional
    // types that are provided during schedule

    /// <summary>
    /// The execution is unrolled.
    /// </summary>
    Unrolled,

    /// <summary>
    /// The loop is vectorized.
    /// </summary>
    Vectorized,

    /// <summary>
    /// The loop is parallelized.
    /// </summary>
    Parallelized,

    /// <summary>
    /// Marks boundary of tensorization intrinsic.
    /// </summary>
    Tensorized,
}
