﻿// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Nncase.Importer;
using Nncase.Importer.TFLite;
using Nncase.IR;

namespace Nncase
{
    /// <summary>
    /// Graph importers.
    /// </summary>
    public static class Importers
    {
        /// <summary>
        /// Import tflite model.
        /// </summary>
        /// <param name="tflite">tflite model stream.</param>
        /// <param name="options">compile options.</param>
        /// <returns>Imported IR module.</returns>
        public static IRModule ImportTFLite(Stream tflite, CompileOptions options)
        {
            var model = new byte[tflite.Length];
            tflite.Read(model);
            var importer = new TFLiteImporter(model);
            return importer.Import(options);
        }

        /// <summary>
        /// Import onnx model.
        /// </summary>
        /// <param name="onnx">onnx model contents.</param>
        /// <param name="options">options.</param>
        /// <returns>Imported IR module.</returns>
        public static IRModule ImportOnnx(Stream onnx, CompileOptions options)
        {
            var model = new byte[onnx.Length];
            onnx.Read(model);
            var importer = new OnnxImporter(model);
            return importer.Import(options);
        }
    }
}
