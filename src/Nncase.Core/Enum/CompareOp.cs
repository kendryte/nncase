// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Nncase;

/// <summary>
/// Comparision operator.
/// </summary>
public enum CompareOp : byte
{
    /// <summary>
    /// a == b.
    /// </summary>
    Equal,

    /// <summary>
    /// a != b.
    /// </summary>
    NotEqual,

    /// <summary>
    /// a &lt; b.
    /// </summary>
    LowerThan,

    /// <summary>
    /// a &lt;= b.
    /// </summary>
    LowerOrEqual,

    /// <summary>
    /// a > b.
    /// </summary>
    GreaterThan,

    /// <summary>
    /// a >= b.
    /// </summary>
    GreaterOrEqual,
}
