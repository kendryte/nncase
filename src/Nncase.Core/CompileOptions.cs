// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using Nncase.Diagnostics;
using Nncase.Quantization;

namespace Nncase;

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
}
