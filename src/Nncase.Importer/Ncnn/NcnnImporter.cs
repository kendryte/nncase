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
using LanguageExt;
using NetFabric.Hyperlinq;
using Nncase.IR;
using Nncase.IR.F;
using Nncase.IR.Math;
using Nncase.IR.NN;
using Nncase.IR.Tensors;
using Math = System.Math;
using Tuple = Nncase.IR.Tuple;

namespace Nncase.Importer.Ncnn;

/// <summary>
/// Ncnn importer.
/// </summary>
public sealed partial class NcnnImporter : BaseImporter
{
    private readonly NcnnGraphImporter _graphImporter;

    /// <summary>
    /// Initializes a new instance of the <see cref="NcnnImporter"/> class.
    /// </summary>
    /// <param name="ncnnParam">Ncnn param stream.</param>
    /// <param name="ncnnBin">Ncnn bin stream.</param>
    /// <param name="compileSession">Compile session.</param>
    public NcnnImporter(Stream ncnnParam, Stream ncnnBin, CompileSession compileSession)
        : base(compileSession)
    {
        _graphImporter = new NcnnGraphImporter(ncnnParam, ncnnBin, compileSession, IRModule);
    }

    protected override BaseGraphImporter CreateMainGraphImporter() => _graphImporter;

    protected override (IEnumerable<Var> Inputs, Dictionary<Var, Expr[]> VarMap) CreateGraphInputs() => _graphImporter.CreateModelInputs();
}
