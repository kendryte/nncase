// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using NetFabric.Hyperlinq;
using Nncase.Diagnostics;
using Nncase.IR;

namespace Nncase;

/// <summary>
/// Importer base.
/// </summary>
public abstract class BaseImporter
{
    /// <summary>
    /// Initializes a new instance of the <see cref="BaseImporter"/> class.
    /// </summary>
    /// <param name="compileSession">Compile session.</param>
    public BaseImporter(CompileSession compileSession)
    {
        CompileSession = compileSession;
        Dumpper = DumpScope.GetCurrent(compileSession).CreateSubDummper("Import", null);
    }

    /// <summary>
    /// Gets compile session.
    /// </summary>
    protected CompileSession CompileSession { get; }

    /// <summary>
    /// Gets dumpper.
    /// </summary>
    protected IDumpper Dumpper { get; }

    protected IRModule IRModule { get; } = new();

    /// <summary>
    /// import the model.
    /// </summary>
    /// <returns>IRModule.</returns>
    public IRModule Import()
    {
        (var inputs, var varMap) = CreateGraphInputs();
        var mainImporter = CreateMainGraphImporter();
        var result = mainImporter.Import();
        AddFunction(mainImporter.Name, inputs.ToArray(), varMap, result.Outputs);
        return IRModule;
    }

    protected abstract BaseGraphImporter CreateMainGraphImporter();

    protected abstract (IEnumerable<Var> Inputs, Dictionary<Var, Expr[]> VarMap) CreateGraphInputs();

    private void AddFunction(string name, Var[] inputs, Dictionary<Var, Expr[]> varMap, Expr body)
    {
        var func = new Function(name, body, inputs, varMap);
        IRModule.Add(func);
        IRModule.Entry = func;
    }
}
