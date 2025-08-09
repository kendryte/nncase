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
    private BaseFunction? _entry;

    /// <summary>
    /// Initializes a new instance of the <see cref="IRModule"/> class.
    /// </summary>
    /// <param name="main">main func.</param>
    public IRModule(BaseFunction main)
    {
        _functions = new() { main };
        main.AddUser(_exprUser);
        Entry = main;
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
        get => _entry;
        set
        {
            if (Functions.IndexOf(value, ReferenceEqualityComparer.Instance) == -1)
            {
                throw new ArgumentException("Entry function must be in the module functions list.");
            }

            if (_entry is not null)
            {
                _entry.IsEntry = false;
            }

            _entry = value;
            if (value is not null)
            {
                value.IsEntry = true;
            }
        }
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
        var old = _functions[index];
        if (old.IsAlive)
        {
            old.ReplaceAllUsesWith(function);
            GC.Collect();
        }

        _functions[index] = function;
        if (object.ReferenceEquals(old, _entry))
        {
            Entry = function;
        }
    }

    /// <summary>
    /// Remove function .
    /// </summary>
    /// <param name="function">function.</param>
    public void Remove(BaseFunction function)
    {
        var index = _functions.FindIndex(x => object.ReferenceEquals(x, function));
        if (index == -1)
        {
            return;
        }

        if (object.ReferenceEquals(function, _entry))
        {
            Entry = null;
        }

        function.RemoveUser(_exprUser);
        _functions.RemoveAt(index);
        GC.Collect();
    }
}
