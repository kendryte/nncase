// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.CommandLine;
using System.CommandLine.Invocation;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Hosting;
using Nncase.CodeGen;
using Nncase.Compiler;
using Nncase.Diagnostics;
using Nncase.IR;
using Nncase.Passes;
using Nncase.Quantization;

namespace Nncase.Cli;

internal enum QuantType
{
    UInt8,
    Int8,
    Int16,
}

internal enum DatasetFormat
{
    Image,
    Raw,
    Pytest,
    Random,
}

/// <summary>
/// Compile command.
/// </summary>
internal sealed class CompileCommand : Command
{
    /// <summary>
    /// Initializes a new instance of the <see cref="CompileCommand"/> class.
    /// </summary>
    public CompileCommand()
        : base("compile")
    {
        InputFile = new Argument<string>("input-file");
        OutputFile = new Argument<string>("output-file");
        InputFormat = new Option<string>(
          aliases: new[] { "-i", "--input-format" },
          description: "input format, e.g. tflite",
          getDefaultValue: () => "tflite");
        DumpFlags = new Option<IEnumerable<DumpFlags>>(
          name: "--dump-flags",
          description: "dump ir flags. \navailable value: None,ImportOps,PassIR,EGraphCost,Rewrite,Calibration,Evaluator,Compile,Tiling,Schedule,CodeGen.")
        {
            AllowMultipleArgumentsPerToken = true,
        };
        DumpDir = new Option<string>(
          name: "--dump-dir",
          description: "dump to directory.",
          getDefaultValue: () => ".");
        QuantType = new Option<QuantType>(
          name: "--quant-type",
          description: $"quant type",
          getDefaultValue: () => Nncase.Cli.QuantType.UInt8);
        WQuantType = new Option<QuantType>(
          name: "--wquant-type",
          description: $"wquant type",
          getDefaultValue: () => Nncase.Cli.QuantType.UInt8);
        Dataset = new Option<string>(
          name: "--dataset",
          description: $"calibration dataset, used in post quantization",
          getDefaultValue: () => string.Empty);
        DatasetFormat = new Option<DatasetFormat>(
          name: "--dataset-format",
          description: $"datset format.",
          getDefaultValue: () => Nncase.Cli.DatasetFormat.Raw);
        ModelQuantMode = new Option<Quantization.ModelQuantMode>(
          name: "--model-quant-mode",
          description: $"model quant mode",
          getDefaultValue: () => Quantization.ModelQuantMode.NoQuant);
        CalibMethod = new Option<Quantization.CalibMethod>(
          name: "--calib-method",
          description: $"model quant options",
          getDefaultValue: () => Quantization.CalibMethod.Kld);
        FixedVars = new Option<IEnumerable<(string, int)>>(
          name: "--fixed-vars",
          description: $"dynamic shape fixed vars, default is empty. \nset by `n:123`",
          parseArgument: result =>
            {
                return result.Tokens.
                    Select(tk => tk.Value.Split(":").ToArray()).
                    Select(tp => (tp[0].Trim(), int.Parse(tp[1].Trim())));
            })
        {
            AllowMultipleArgumentsPerToken = true,
        };
        AddArgument(InputFile);
        AddArgument(OutputFile);
        AddGlobalOption(InputFormat);
        AddGlobalOption(DumpFlags);
        AddGlobalOption(DumpDir);
        AddGlobalOption(QuantType);
        AddGlobalOption(WQuantType);
        AddGlobalOption(Dataset);
        AddGlobalOption(DatasetFormat);
        AddGlobalOption(ModelQuantMode);
        AddGlobalOption(CalibMethod);
        AddGlobalOption(FixedVars);
    }

    public Argument<string> InputFile { get; }

    public Argument<string> OutputFile { get; }

    public Option<string> InputFormat { get; }

    public Option<IEnumerable<DumpFlags>> DumpFlags { get; }

    public Option<string> DumpDir { get; }

    public Option<QuantType> QuantType { get; }

    public Option<QuantType> WQuantType { get; }

    public Option<string> Dataset { get; }

    public Option<DatasetFormat> DatasetFormat { get; }

    public Option<ModelQuantMode> ModelQuantMode { get; }

    public Option<CalibMethod> CalibMethod { get; }

    public Option<IEnumerable<(string Name, int Value)>> FixedVars { get; }
}
