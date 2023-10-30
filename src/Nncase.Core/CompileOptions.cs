// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using Nncase.Diagnostics;
using Nncase.Quantization;

namespace Nncase;

public enum InputType : int
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
    /// float32.
    /// </summary>
    Float32,
}

/// <summary>
/// Compile options.
/// </summary>
public sealed record CompileOptions
{
    /// <summary>
    /// Gets or sets input file.
    /// </summary>
    public string InputFile { get; set; } = "<stream>";

    /// <summary>
    /// Gets or sets the import model format.
    /// </summary>
    public string InputFormat { get; set; } = "onnx";

    /// <summary>
    /// Gets or sets the dump flags.
    /// </summary>
    public DumpFlags DumpFlags { get; set; } = DumpFlags.None;

    /// <summary>
    /// Gets or sets the dump directory.
    /// </summary>
    public string DumpDir { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets quant options.
    /// </summary>
    public QuantizeOptions QuantizeOptions { get; set; } = QuantizeOptions.CreateNoQuant();

    /// <summary>
    /// Gets or sets a value indicating whether gets or sets the preprocess.
    /// </summary>
    public bool PreProcess { get; set; }

    /// <summary>
    /// Gets or sets the input layout.
    /// </summary>
    public string InputLayout { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets the output type.
    /// </summary>
    public string OutputLayout { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets the input type.
    /// </summary>
    public InputType InputType { get; set; } = InputType.Float32;

    /// <summary>
    /// Gets or sets the input shape.
    /// </summary>
    public int[] InputShape { get; set; } = Array.Empty<int>();

    /// <summary>
    /// Gets or sets the input range.
    /// </summary>
    public float[] InputRange { get; set; } = Array.Empty<float>();

    /// <summary>
    /// Gets or sets a value indicating whether gets or sets the swapRB.
    /// </summary>
    public bool SwapRB { get; set; }

    /// <summary>
    /// Gets or sets the letterbox_value.
    /// </summary>
    public float LetterBoxValue { get; set; }

    /// <summary>
    /// Gets or sets the mean.
    /// </summary>
    public float[] Mean { get; set; } = Array.Empty<float>();

    /// <summary>
    /// Gets or sets the std.
    /// </summary>
    public float[] Std { get; set; } = Array.Empty<float>();

    /// <summary>
    /// Gets or sets the std.
    /// </summary>
    public string ModelLayout { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets quant options.
    /// </summary>
    public ShapeBucketOptions ShapeBucketOptions { get; set; } = ShapeBucketOptions.Default;

    /// <summary>
    /// Gets or sets the target compile options.
    /// </summary>
    public ITargetCompileOptions TargetCompileOptions { get; set; } = null!;
}
