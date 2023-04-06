// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Nncase.Quantization;

/// <summary>
/// Model Quant Mode.
/// </summary>
public enum ModelQuantMode : int
{
    /// <summary>
    /// no quant.
    /// </summary>
    NoQuant,

    /// <summary>
    /// use ptq.
    /// </summary>
    UsePTQ,

    /// <summary>
    /// use qat.
    /// </summary>
    UseQAT,
}

/// <summary>
/// Quant Type.
/// </summary>
public enum QuantType : int
{
    /// <summary>
    /// uint8.
    /// </summary>
    Uint8,

    /// <summary>
    /// int8.
    /// </summary>
    Int8,

    /// <summary>
    /// int16.
    /// </summary>
    Int16,
}

/// <summary>
/// Fine Tune Weights Method.
/// </summary>
public enum FineTuneWeightsMethod : int
{
    /// <summary>
    /// no fine tune weights.
    /// </summary>
    NoFineTuneWeights,

    /// <summary>
    /// use sqaunt.
    /// </summary>
    UseSquant,

    /// <summary>
    /// use adaround.
    /// </summary>
    UseAdaRound,
}

/// <summary>
/// Calibration Method.
/// </summary>
public enum CalibMethod
{
    /// <summary>
    /// no clip.
    /// </summary>
    NoClip,

    /// <summary>
    /// kld.
    /// </summary>
    Kld,

    /// <summary>
    /// use random data only for test.
    /// </summary>
    Random,
}

/// <summary>
/// quantize options.
/// </summary>
public class QuantizeOptions
{
    /// <summary>
    /// Gets or sets calibrationDataset.
    /// </summary>
    public ICalibrationDatasetProvider? CalibrationDataset { get; set; }

    /// <summary>
    /// Gets or sets calibMethod.
    /// </summary>
    public CalibMethod CalibrationMethod { get; set; } = CalibMethod.NoClip;

    /// <summary>
    /// Gets or sets a value indicating whether enable the Auto bind quant method.
    /// </summary>
    public bool BindQuantMethod { get; set; }

    /// <summary>
    /// Gets or sets a value indicating whether enable squant to fine tune weights.
    /// </summary>
    public bool UseSquant { get; set; }

    /// <summary>
    /// Gets or sets a value indicating whether enable adaround to fine tune weights.
    /// </summary>
    public bool UseAdaRound { get; set; }

    /// <summary>
    /// Gets or sets quant type.
    /// </summary>
    public DataType QuantType { get; set; } = DataTypes.UInt8;

    /// <summary>
    /// Gets or sets weights quant type.
    /// </summary>
    public DataType WQuantType { get; set; } = DataTypes.UInt8;

    /// <summary>
    /// Gets or sets model quant mode.
    /// </summary>
    public ModelQuantMode ModelQuantMode { get; set; } = ModelQuantMode.NoQuant;

    /// <summary>
    /// Gets or sets import config file.
    /// </summary>
    public string ImportConfigFile { get; set; } = string.Empty;

    /// <summary>
    /// Creates no quantization options.
    /// </summary>
    /// <returns>No quant options.</returns>
    public static QuantizeOptions CreateNoQuant() => new QuantizeOptions();
}
