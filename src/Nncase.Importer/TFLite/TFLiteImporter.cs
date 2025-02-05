// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using FlatBuffers;
using NetFabric.Hyperlinq;
using Nncase.IR;
using Nncase.IR.F;
using Nncase.IR.Math;
using Nncase.IR.Tensors;
using Math = System.Math;
using Tuple = Nncase.IR.Tuple;

namespace Nncase.Importer.TFLite;

/// <summary>
/// TFLite importer.
/// </summary>
public sealed partial class TFLiteImporter : BaseImporter
{
    private readonly tflite.Model _model;
    private readonly TFLiteGraphImporter _mainGraphImporter;

    /// <summary>
    /// Initializes a new instance of the <see cref="TFLiteImporter"/> class.
    /// </summary>
    /// <param name="tfliteModel">TFLite model bytes.</param>
    /// <param name="compileSession">Compile session.</param>
    public TFLiteImporter(byte[] tfliteModel, CompileSession compileSession)
        : base(compileSession)
    {
        _model = tflite.Model.GetRootAsModel(new ByteBuffer(tfliteModel));
        if (!tflite.Model.ModelBufferHasIdentifier(_model.ByteBuffer))
        {
            throw new InvalidDataException("Invalid tflite model file.");
        }

        _mainGraphImporter = new TFLiteGraphImporter(_model, _model.Subgraphs(0)!.Value, CompileSession, IRModule);
    }

    protected override BaseGraphImporter CreateMainGraphImporter() => _mainGraphImporter;

    protected override (IEnumerable<Var> Inputs, Dictionary<Var, Expr[]> VarMap) CreateGraphInputs() => _mainGraphImporter.CreateModelInputs();
}
