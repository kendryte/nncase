// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using Nncase.Quantization;

namespace Nncase;

/// <summary>
/// CompileOptions.
/// </summary>
public sealed class CompileOptions
{
    /// <summary>
    /// Initializes a new instance of the <see cref="CompileOptions"/> class.
    /// copy ctor.
    /// </summary>
    public CompileOptions(CompileOptions other)
    {
        InputFile = other.InputFile;
        InputFormat = other.InputFormat;
        Target = other.Target;
        DumpLevel = other.DumpLevel;
        DumpDir = other.DumpDir;
        ModelQuantMode = other.ModelQuantMode;
        QuantType = other.QuantType;
        WQuantType = other.WQuantType;
        OutputFile = other.OutputFile;
    }

    /// <summary>
    /// Initializes a new instance of the <see cref="CompileOptions"/> class.
    /// CompileOptions.
    /// </summary>
    public CompileOptions()
    {
        InputFile = string.Empty;
        InputFormat = string.Empty;
        Target = string.Empty;
        DumpLevel = -1;
        DumpDir = string.Empty;
        ModelQuantMode = ModelQuantMode.NoQuant;
        QuantType = DataTypes.UInt8;
        WQuantType = DataTypes.UInt8;
        OutputFile = string.Empty;
    }

    /// <summary>
    /// Initializes a new instance of the <see cref="CompileOptions"/> class.
    /// init.
    /// </summary>
    /// <param name="modelQuantMode"></param>
    public CompileOptions(ModelQuantMode modelQuantMode)
    {
        InputFile = string.Empty;
        InputFormat = string.Empty;
        Target = string.Empty;
        DumpLevel = -1;
        DumpDir = string.Empty;
        ModelQuantMode = modelQuantMode;
        QuantType = DataTypes.UInt8;
        WQuantType = DataTypes.UInt8;
        OutputFile = string.Empty;
        QuantizeOptions = QuantizeOptions;
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
    public DataType QuantType { get; set; }

    /// <inheritdoc/>
    public DataType WQuantType { get; set; }

    /// <inheritdoc/>
    public string OutputFile { get; set; }

    /// <inheritdoc/>
    public ModelQuantMode ModelQuantMode { get; set; }

    /// <inheritdoc/>
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

// /// <summary>
//     /// Gets or sets output file.
//     /// </summary>
//     public string OutputFile { get; set; }

// /// <summary>
//     /// Gets or sets the import model format.
//     /// </summary>
//     public string InputFormat { get; set; }

// /// <summary>
//     /// Gets or sets target.
//     /// </summary>
//     public string Target { get; set; }

// /// <summary>
//     /// Gets or sets the dump level.
//     /// </summary>
//     public int DumpLevel { get; set; }

// /// <summary>
//     /// Gets or sets the dump directory.
//     /// </summary>
//     public string DumpDir { get; set; }

// /// <summary>
//     /// weather use ptq
//     /// </summary>
//     public bool UsePTQ { get; set; }

// /// <summary>
//     /// Gets or sets quant type
//     /// </summary>
//     public DataType QuantType { get; set; }

// /// <summary>
//     /// Gets or sets quant mode
//     /// </summary>
//     public QuantMode QuantMode { get; set; }
// }
