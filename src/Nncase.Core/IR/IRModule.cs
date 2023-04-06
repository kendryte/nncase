// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Reactive;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading.Tasks;
using Nncase.IR;

namespace Nncase.IR;

/// <summary>
/// Module.
/// </summary>
public sealed class IRModule
{
    private readonly List<BaseFunction> _functions;
    private readonly ExprUser _exprUser = new();
    private int? _entryIndex;

    /// <summary>
    /// Initializes a new instance of the <see cref="IRModule"/> class.
    /// </summary>
    /// <param name="main">main func.</param>
    public IRModule(BaseFunction main)
    {
        _functions = new() { main };
        _entryIndex = 0;
        main.AddUser(_exprUser);
    }

    /// <summary>
    /// Initializes a new instance of the <see cref="IRModule"/> class.
    /// the default IrModule ctor.
    /// </summary>
    public IRModule()
    {
        _functions = new();
    }

    /// <summary>
    /// Gets functions.
    /// </summary>
    public IReadOnlyList<BaseFunction> Functions => _functions;

    /// <summary>
    /// Gets or sets entry function.
    /// </summary>
    public BaseFunction? Entry
    {
        get => _entryIndex.HasValue ? _functions[_entryIndex.Value] : null;
        set => _entryIndex = value != null ? _functions.FindIndex(x => object.ReferenceEquals(x, value)) : null;
    }

    /// <summary>
    /// Add function.
    /// </summary>
    /// <param name="function">Callable to add.</param>
    public void Add(BaseFunction function)
    {
        CompilerServices.InferenceType(function);
        _functions.Add(function);
        function.AddUser(_exprUser);
    }

    /// <summary>
    /// Replace the function defination.
    /// </summary>
    /// <param name="index">function index.</param>
    /// <param name="function">the entry function defination.</param>
    public void Replace(int index, BaseFunction function)
    {
        CompilerServices.InferenceType(function);
        ref var old = ref CollectionsMarshal.AsSpan(_functions)[index];
        if (old.IsAlive)
        {
            old.ReplaceAllUsesWith(function);
            old.DisposeIfNoUsers();
        }

        old = function;
    }
}
