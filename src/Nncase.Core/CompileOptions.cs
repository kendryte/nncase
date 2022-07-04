// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using Nncase.Quantization;

namespace Nncase;

/// <summary>
/// CompileOptions
/// </summary>
public sealed class CompileOptions
{

    /// <summary>
    /// copy ctor
    /// </summary>
    public CompileOptions(CompileOptions other)
    {
        InputFile = other.InputFile;
        InputFormat = other.InputFormat;
        Target = other.Target;
        DumpLevel = other.DumpLevel;
        DumpDir = other.DumpDir;
        UsePTQ = other.UsePTQ;
        QuantType = other.QuantType;
        QuantMode = other.QuantMode;
        OutputFile = other.OutputFile;

    }

    /// <summary>
    /// CompileOptions
    /// </summary>
    public CompileOptions()
    {
        InputFile = string.Empty;
        InputFormat = string.Empty;
        Target = string.Empty;
        DumpLevel = -1;
        DumpDir = string.Empty;
        UsePTQ = false;
        QuantType = DataTypes.Int8;
        QuantMode = QuantMode.UnsignedMode;
        OutputFile = string.Empty;
    }

    /// <summary>
    /// init 
    /// </summary>
    /// <param name="usePtq"></param>
    public CompileOptions(bool usePtq)
    {
        InputFile = string.Empty;
        InputFormat = string.Empty;
        Target = string.Empty;
        DumpLevel = -1;
        DumpDir = string.Empty;
        UsePTQ = usePtq;
        QuantType = DataTypes.Int8;
        QuantMode = QuantMode.UnsignedMode;
        OutputFile = string.Empty;
    }

    /// <inheritdoc/>
    public string InputFile { get; set; }

    /// <inheritdoc/>
    public string InputFormat { get; set; }

    /// <inheritdoc/>
    public string Target { get; set; }

    /// <inheritdoc/>
    public int DumpLevel { get; set; }

    /// <inheritdoc/>
    public string DumpDir { get; set; }

    /// <inheritdoc/>
    public bool UsePTQ { get; set; }

    /// <inheritdoc/>
    public DataType QuantType { get; set; }

    /// <inheritdoc/>
    public QuantMode QuantMode { get; set; }

    /// <inheritdoc/>
    public string OutputFile { get; set; }

    public QuantizeOptions? QuantizeOptions { get; set; }
}

// /// <summary>
// /// Options of compile command.
// /// </summary>
// public interface CompileOptions
// {
//     /// <summary>
//     /// Gets or sets input file.
//     /// </summary>
//     public string InputFile { get; set; }

//     /// <summary>
//     /// Gets or sets output file.
//     /// </summary>
//     public string OutputFile { get; set; }

//     /// <summary>
//     /// Gets or sets the import model format.
//     /// </summary>
//     public string InputFormat { get; set; }

//     /// <summary>
//     /// Gets or sets target.
//     /// </summary>
//     public string Target { get; set; }

//     /// <summary>
//     /// Gets or sets the dump level.
//     /// </summary>
//     public int DumpLevel { get; set; }

//     /// <summary>
//     /// Gets or sets the dump directory.
//     /// </summary>
//     public string DumpDir { get; set; }

//     /// <summary>
//     /// weather use ptq
//     /// </summary>
//     public bool UsePTQ { get; set; }

//     /// <summary>
//     /// Gets or sets quant type
//     /// </summary>
//     public DataType QuantType { get; set; }

//     /// <summary>
//     /// Gets or sets quant mode
//     /// </summary>
//     public QuantMode QuantMode { get; set; }
// }

