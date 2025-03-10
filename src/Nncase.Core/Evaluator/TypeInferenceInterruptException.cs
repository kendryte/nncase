// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Nncase.IR;

namespace Nncase.Evaluator;

/// <summary>
/// Interrupt type inference.
/// </summary>
public sealed class TypeInferenceInterruptException : ApplicationException
{
    /// <summary>
    /// Initializes a new instance of the <see cref="TypeInferenceInterruptException"/> class.
    /// </summary>
    /// <param name="reasonType">Reason type.</param>
    public TypeInferenceInterruptException(IRType reasonType)
        : this(reasonType, null)
    {
    }

    /// <summary>
    /// Initializes a new instance of the <see cref="TypeInferenceInterruptException"/> class.
    /// </summary>
    /// <param name="reasonType">Reason type.</param>
    /// <param name="innerException">Inner exception.</param>
    public TypeInferenceInterruptException(IRType reasonType, Exception? innerException)
        : base(null, innerException)
    {
        ReasonType = reasonType;
    }

    /// <summary>
    /// Gets reason type.
    /// </summary>
    public IRType ReasonType { get; }
}
