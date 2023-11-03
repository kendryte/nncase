// Copyright (c) Canaan Inc. All rights reserved.
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
        using var fileStream = File.OpenRead(fileName);
        switch (modelFormat.ToUpperInvariant())
        {
            case "TFLITE":
                return await compiler.ImportTFLiteModuleAsync(fileStream);
            case "ONNX":
                return await compiler.ImportOnnxModuleAsync(fileStream);
            case "NCNN":
                {
                    using var binStream = isBenchmarkOnly ? (Stream)new ZeroStream() : File.OpenRead(Path.ChangeExtension(fileName, "bin"));
                    return await compiler.ImportNcnnModuleAsync(fileStream, binStream);
                }

            default:
                throw new NotSupportedException($"Unsupported model format: {modelFormat}");
        }
    }
}
