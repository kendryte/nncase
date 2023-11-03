// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Nncase.Importer;
using Nncase.Importer.Ncnn;
using Nncase.Importer.TFLite;
using Nncase.IR;

namespace Nncase;

/// <summary>
/// Graph importers.
/// </summary>
public static class Importers
{
    /// <summary>
    /// Import tflite model.
    /// </summary>
    /// <param name="tflite">tflite model stream.</param>
    /// <param name="compileSession">compile session.</param>
    /// <returns>Imported IR module.</returns>
    public static IRModule ImportTFLite(Stream tflite, CompileSession compileSession)
    {
        compileSession.CompileOptions.ModelLayout = "NHWC";
        var model = new byte[tflite.Length];
        tflite.Read(model);
        var importer = new TFLiteImporter(model, compileSession);
        return importer.Import();
    }

    /// <summary>
    /// Import onnx model.
    /// </summary>
    /// <param name="onnx">onnx model contents.</param>
    /// <param name="compileSession">compile session.</param>
    /// <returns>Imported IR module.</returns>
    public static IRModule ImportOnnx(Stream onnx, CompileSession compileSession)
    {
        compileSession.CompileOptions.ModelLayout = "NCHW";
        var importer = new OnnxImporter(onnx, compileSession);
        return importer.Import();
    }

    /// <summary>
    /// Import ncnn model.
    /// </summary>
    /// <param name="ncnnParam">Ncnn param stream.</param>
    /// <param name="ncnnBin">Ncnn bin stream.</param>
    /// <param name="compileSession">compile session.</param>
    /// <returns>Imported IR module.</returns>
    public static IRModule ImportNcnn(Stream ncnnParam, Stream ncnnBin, CompileSession compileSession)
    {
        compileSession.CompileOptions.ModelLayout = "NCHW";
        var importer = new NcnnImporter(ncnnParam, ncnnBin, compileSession);
        return importer.Import();
    }
}
