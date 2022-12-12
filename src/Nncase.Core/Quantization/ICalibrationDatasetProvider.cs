// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Nncase.IR;

namespace Nncase.Quantization;

/// <summary>
/// Calibration dataset provider.
/// </summary>
public interface ICalibrationDatasetProvider
{
    /// <summary>
    /// Gets the count of samples.
    /// </summary>
    int? Count { get; }

    /// <summary>
    /// Gets the samples.
    /// </summary>
    IAsyncEnumerable<IReadOnlyDictionary<Var, IValue>> Samples { get; }
}
