// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.Diagnostics;
using System.IO;
using System.Linq;
using Google.Protobuf;
using Google.Protobuf.Collections;
using LanguageExt;
using NetFabric.Hyperlinq;
using Nncase.IR;
using Nncase.IR.Tensors;
using Onnx;
using Tuple = Nncase.IR.Tuple;

namespace Nncase.Importer;

public sealed partial class OnnxImporter : BaseImporter
{
    private readonly ModelProto _model;
    private readonly Dictionary<string, long> _opSetMap;
    private readonly OnnxGraphImporter _mainGraphImporter;

    /// <summary>
    /// Initializes a new instance of the <see cref="OnnxImporter"/> class.
    /// </summary>
    /// <param name="onnxModel">Onnx model stream.</param>
    /// <param name="compileSession">Compile session.</param>
    public OnnxImporter(Stream onnxModel, CompileSession compileSession)
        : base(compileSession)
    {
        _opSetMap = new Dictionary<string, long>();
        _model = ModelProto.Parser.ParseFrom(new CodedInputStream(onnxModel, true));

        foreach (var opSet in _model.OpsetImport)
        {
            _opSetMap.Add(opSet.Domain, opSet.Version);
        }

        _mainGraphImporter = new OnnxGraphImporter(null, _model.Graph, _opSetMap, CompileSession, IRModule);
    }

    protected override BaseGraphImporter CreateMainGraphImporter() => _mainGraphImporter;

    /// <inheritdoc/>
    protected override (IEnumerable<Var> Inputs, Dictionary<Var, Expr[]> VarMap) CreateGraphInputs() => _mainGraphImporter.CreateModelInputs(_model);
}
