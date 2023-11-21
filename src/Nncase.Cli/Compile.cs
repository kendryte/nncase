// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.CommandLine;
using System.Linq;
using Nncase.Diagnostics;
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
        PreProcess = new Option<bool>(
            name: "--pre-process",
            description: "whether enable pre process",
            getDefaultValue: () => false);
        InputLayout = new Option<string>(
            name: "--input-layout",
            description: "the model input data layout",
            getDefaultValue: () => string.Empty).FromAmong("NCHW", "NHWC");
        OutputLayout = new Option<string>(
            name: "--output-layout",
            description: "the model output data layout.",
            getDefaultValue: () => string.Empty).FromAmong("NCHW", "NHWC");
        InputType = new Option<InputType>(
            name: "--input-type",
            description: "the model input data value type, default is Float32",
            getDefaultValue: () => Nncase.InputType.Float32);
        InputShape = new Option<IEnumerable<int>>(
            name: "--input-shape",
            description: "the model input data shape. eg. `--input-shape 1 2 3 4`",
            getDefaultValue: Array.Empty<int>)
        {
            AllowMultipleArgumentsPerToken = true,
        };
        InputRange = new Option<IEnumerable<float>>(
            name: "--input-range",
            description: "the model input data value range. eg `--input-range -100.3 200.4`",
            getDefaultValue: Array.Empty<float>)
        {
            AllowMultipleArgumentsPerToken = true,
        };
        SwapRB = new Option<bool>(
            name: "--swap-rb",
            description: "whether swap the model input data channel, like cv2.BGRtoRGB(im)",
            getDefaultValue: () => false);
        LetterBoxValue = new Option<float>(
            name: "--letter-box-value",
            description: "letterbox fill value",
            getDefaultValue: () => 0.0f);
        Mean = new Option<IEnumerable<float>>(
            name: "--mean",
            description: "the model input data mean, default []",
            getDefaultValue: Array.Empty<float>)
        {
            AllowMultipleArgumentsPerToken = true,
        };
        Std = new Option<IEnumerable<float>>(
            name: "--std",
            description: "the model input data std, default []",
            getDefaultValue: Array.Empty<float>)
        {
            AllowMultipleArgumentsPerToken = true,
        };
        ModelLayout = new Option<string>(
            name: "--model-layout",
            description: "the model's input layout.",
            getDefaultValue: () => string.Empty).FromAmong("NCHW", "NHWC");
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
        AddGlobalOption(PreProcess);
        AddGlobalOption(InputLayout);
        AddGlobalOption(OutputLayout);
        AddGlobalOption(InputType);
        AddGlobalOption(InputShape);
        AddGlobalOption(InputRange);
        AddGlobalOption(SwapRB);
        AddGlobalOption(LetterBoxValue);
        AddGlobalOption(Mean);
        AddGlobalOption(Std);
        AddGlobalOption(ModelLayout);
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

    public Option<bool> PreProcess { get; }

    public Option<string> InputLayout { get; }

    public Option<string> OutputLayout { get; }

    public Option<InputType> InputType { get; }

    public Option<IEnumerable<int>> InputShape { get; }

    public Option<IEnumerable<float>> InputRange { get; }

    public Option<bool> SwapRB { get; }

    public Option<float> LetterBoxValue { get; }

    public Option<IEnumerable<float>> Mean { get; }

    public Option<IEnumerable<float>> Std { get; }

    public Option<string> ModelLayout { get; }
}
