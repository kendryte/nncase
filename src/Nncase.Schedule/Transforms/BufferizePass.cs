// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading.Tasks;
using Nncase.IR;
using Nncase.IR.Affine;
using Nncase.IR.Buffers;

namespace Nncase.Passes.Transforms;

public sealed class BufferizePass : ModulePass
{
    protected override Task<IRModule> RunCoreAsync(IRModule input, RunPassContext context)
    {
        var funcs = input.Functions.Count;
        for (int i = 0; i < funcs; i++)
        {
            if (input.Functions[i] is Function function)
            {
                var newFunc = Bufferize(function, input);
                input.Replace(i, newFunc);
            }
        }

        return Task.FromResult(input);
    }

    private BaseFunction Bufferize(Function function, IRModule module)
    {
        // 1. Merge grids into new PrimFunctions
        throw new NotImplementedException();
    }

    private sealed class GraphMerger
    {
        private readonly Function _function;

        public GraphMerger(Function function)
        {
            _function = function;
        }

        public void Merge(IRModule module)
        {
            CreateRegions();
        }

        private void CreateRegions()
        {
        }

        private sealed class Region
        {
            private readonly HashSet<Expr> _nodesSet = new(ReferenceEqualityComparer.Instance);
            private readonly HashSet<Expr> _inputs = new(ReferenceEqualityComparer.Instance);
            private readonly HashSet<Expr> _outputs = new(ReferenceEqualityComparer.Instance);
            private readonly Dictionary<Expr, int> _nodeOutputUsers = new(ReferenceEqualityComparer.Instance);

            public string ModuleKind { get; private set; } = string.Empty;

            public List<Expr> Nodes { get; } = new();

            public bool Add(Expr node)
            {
                throw new NotImplementedException();
#if false
                if (_nodesSet.Add(node))
                {
                    Nodes.Add(node);
                    _outputs.Add(node);
                    _nodeOutputUsers.Add(node, node.Users.Count);

                    if (_outputs.Contains(input))
                    {
                        // Remove region output if no outer users
                        if (--_nodeOutputUsers[input] == 0)
                        {
                            _outputs.Remove(input);
                            _nodeOutputUsers.Remove(input);
                        }
                    }

                    foreach (var input in node.Operands)
                    {
                        _inputs.Add(input);
                    }

                    foreach (var input in _inputs)
                    {
                    }
#endif
            }
        }
    }
}
