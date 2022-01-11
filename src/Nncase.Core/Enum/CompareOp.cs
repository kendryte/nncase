// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Nncase
{
    /// <summary>
    /// compare op enum
    /// </summary>
    public enum CompareOp
    {
        /// <summary>
        /// a == b
        /// </summary>
        EQ,
        /// <summary>
        /// a != b
        /// </summary>
        NE,
        /// <summary>
        /// a < b
        /// </summary>
        LT,
        /// <summary>
        /// a <= b
        /// </summary>
        LE,
        /// <summary>
        /// a > b
        /// </summary>
        GT,
        /// <summary>
        /// a >= b
        /// </summary>
        GE
    }
}