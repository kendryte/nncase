// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
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
    private int? _entryIndex;

    /// <summary>
    /// Initializes a new instance of the <see cref="IRModule"/> class.
    /// </summary>
    /// <param name="main">main func.</param>
    public IRModule(BaseFunction main)
    {
        _functions = new() { main };
        _entryIndex = 0;
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
        _functions.Add(function);
    }

    /// <summary>
    /// Replace the function defination.
    /// </summary>
    /// <param name="index">function index.</param>
    /// <param name="function">the entry function defination.</param>
    public void Replace(int index, BaseFunction function)
    {
        var old = _functions[index];

        var replacer = new FunctionReplacer(old, function);
        for (int i = 0; i < _functions.Count; i++)
        {
            replacer.Visit(_functions[i]);
        }

        for (int i = 0; i < _functions.Count; i++)
        {
            var originFunc = _functions[i];
            if (replacer.ExpressionMemo.TryGetValue(originFunc, out var replace))
            {
                _functions[i] = (BaseFunction)replace;
            }
        }
    }

    /// <summary>
    /// Replace the function call dependencer.
    /// </summary>
    private sealed class FunctionReplacer : DeepExprMutator
    {
        private readonly BaseFunction _original;
        private readonly BaseFunction _replace;

        public FunctionReplacer(BaseFunction original, BaseFunction replace)
        {
            _original = original;
            _replace = replace;
        }

        public override Expr DefaultMutateLeaf(Expr expr)
        {
            if (expr is BaseFunction baseFunction
                && object.ReferenceEquals(baseFunction, _original))
            {
                return _replace;
            }

            return expr;
        }
    }
}
