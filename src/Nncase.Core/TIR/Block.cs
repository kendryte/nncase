// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Nncase.IR;

namespace Nncase.TIR;

/// <summary>
/// A block is a basic schedule unit in TIR.
/// <remarks>
/// Block's body is parameterized by iter vars.
/// </remarks>
/// <code>
///   with T.block(name):
///   v0 = T.axis.S(domain, value0)
///   v1 = T.axis.R(domain, value1)
///   ...
///   T.reads([buffer0[start:end, ...], ...])
///   T.writes([buffer1[start:end, ...], ...])
///   T.where(predicate)
///   buffer2 = T.alloc_buffer(shape, dtype)
///   buffer3 = T.match_buffer(source_buffer[start:end, ...])
///   T.attr({attr_key: attr_value, ...})
///   with T.init():
///      init body
///    body
/// </code>
/// </summary>
/// <param name="Name"> The name_hint of the block.</param>
/// <param name="Body"> block body. </param>
/// <param name="InitBody">the Block init statement.</param>
/// <param name="IterVars">The List Exprs contain the IterVars.</param>
/// <param name="Reads">The read buffer regions of the block.</param>
/// <param name="Writes">The write buffer regions of the block.</param>
/// <param name="AllocBuffers">The buffer allocated in the block.</param>
/// <param name="Predicate">The predicate of the block realization, the block will only be executed when the predicate is true.</param>
public sealed record Block(string Name, Sequential Body, Sequential InitBody,
                            IRArray<IterVar> IterVars,
                            IRArray<BufferRegion> Reads,
                            IRArray<BufferRegion> Writes,
                            IRArray<TIR.Buffer> AllocBuffers, Expr Predicate) : Expr
{
    /// <summary>
    /// Initializes a new instance of the <see cref="Block"/> class.
    /// </summary>
    /// <param name="name">block name.</param>
    public Block(string name)
        : this(name, new(), new(), new(), new(), new(), new(), true)
    {
    }
}
