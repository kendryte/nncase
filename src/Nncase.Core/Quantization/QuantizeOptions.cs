// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Nncase.Quantization;

/// <summary>
/// Model Quant Mode
/// </summary>
public enum ModelQuantMode : int
{
    /// <summary>
    /// no quant
    /// </summary>
    NoQuant,
    /// <summary>
    /// use ptq
    /// </summary>
    UsePTQ,
    /// <summary>
    /// use qat
    /// </summary>
    UseQAT
}

;

/// <summary>
/// Calibration Method
/// </summary>
public enum CalibMethod
{
    /// <summary>
    /// no clip
    /// </summary>
    NoClip,
    /// <summary>
    /// kld
    /// </summary>
    Kld,
    /// <summary>
    /// use random data only for test
    /// </summary>
    Random
}

/// <summary>
/// quantize options
/// </summary>
public class QuantizeOptions
{
    /// <summary>
    /// CalibrationDataset
    /// </summary>
    public ICalibrationDatasetProvider? CalibrationDataset { get; set; }
    /// <summary>
    /// CalibMethod
    /// </summary>
    public CalibMethod CalibrationMethod { get; set; } = CalibMethod.NoClip;

    /// <summary>
    /// Enable the Auto bind quant method.
    /// </summary>
    public bool BindQuantMethod { get; set; } = false;

    /// <summary>
    /// Enable squant to fine tune weights.
    /// </summary>
    public bool UseSquant { get; set; } = false;

    /// <summary>
    /// Enable adaround to fine tune weights.
    /// </summary>
    public bool UseAdaRound { get; set; } = false;
}
