﻿// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Nncase.IO;
using Nncase.IR;

namespace Nncase;

public static class CompilerExtensions
{
    public static async Task<IRModule> ImportModuleAsync(this ICompiler compiler, string modelFormat, string fileName, bool isBenchmarkOnly = false)
    {
        switch (modelFormat.ToUpperInvariant())
        {
            case "TFLITE":
                {
                    using var fileStream = File.OpenRead(fileName);
                    return await compiler.ImportTFLiteModuleAsync(fileStream);
                }

            case "ONNX":
                {
                    using var fileStream = File.OpenRead(fileName);
                    return await compiler.ImportOnnxModuleAsync(fileStream);
                }

            case "PARAM":
                {
                    using var fileStream = File.OpenRead(fileName);
                    using var binStream = isBenchmarkOnly ? (Stream)new ZeroStream() : File.OpenRead(Path.ChangeExtension(fileName, "bin"));
                    return await compiler.ImportNcnnModuleAsync(fileStream, binStream);
                }

            case "HUGGINGFACE":
                return await compiler.ImportHuggingFaceModuleAsync(fileName, new ImportOptions()
                {
                    HuggingFaceOptions = CompileSessionScope.Current!.CompileOptions.HuggingFaceOptions,
                });

            default:
                throw new NotSupportedException($"Unsupported model format: {modelFormat}");
        }
    }
}
