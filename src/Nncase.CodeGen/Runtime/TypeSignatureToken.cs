// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Nncase.IR;

namespace Nncase.Runtime;

/// <summary>
/// Type signature token.
/// </summary>
/// <remarks>[WARN] Sync with Native/include/nncase/runtime/type_serializer.h.</remarks>
public enum TypeSignatureToken : byte
{
    /// <summary>
    /// <see cref="InvalidType"/>.
    /// </summary>
    Invalid,

    /// <summary>
    /// <see cref="AnyType"/>.
    /// </summary>
    Any,

    /// <summary>
    /// <see cref="TensorType"/>.
    /// </summary>
    Tensor,

    /// <summary>
    /// <see cref="TupleType"/>.
    /// </summary>
    Tuple,

    /// <summary>
    /// <see cref="CallableType"/>.
    /// </summary>
    Callable,

    /// <summary>
    /// End of type signature.
    /// </summary>
    End = 0xFF,
}
