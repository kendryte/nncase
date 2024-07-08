// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using NetFabric.Hyperlinq;
using Nncase.IR;
using Nncase.IR.Affine;
using Nncase.Schedule;

namespace Nncase.Passes.Transforms;

public sealed class AutoTilePass : ModulePass
{
    public AutoTilePass(CompileOptions compileOptions)
    {
        CompileOptions = compileOptions;
    }

    public CompileOptions CompileOptions { get; }

    protected override Task<IRModule> RunCoreAsync(IRModule input, RunPassContext context)
    {
        var funcNums = input.Functions.Count;
        for (int i = 0; i < funcNums; i++)
        {
            // top sorted
            var collects = ExprCollector.Collect(input.Functions[i]).OfType<Grid>().ToArray();
            var worklists = GatherWorkLists(collects);
            var post = input.Functions[i];
            for (int j = 0; j < worklists.Count; j++)
            {
                var rootGrid = worklists[j].First();
                var rewriter = new AutoTileRewriter(rootGrid, CompileOptions);
                post = (BaseFunction)rewriter.Rewrite(post);

                // if (rewriter.IsMutated)
                // {
                //     input.Add(rewriter.Wrapper);
                //     input.Add(rewriter.PrimFunc);
                // }
            }

            input.Replace(i, post);
        }

        return Task.FromResult(input);
    }

    private List<List<Grid>> GatherWorkLists(IReadOnlyList<Grid> collects)
    {
        List<List<Grid>> workLists = new();
        for (int i = collects.Count - 1; i >= 0;)
        {
            // start find single input.
            if (collects[i] is Grid sinkNode)
            {
                var current = sinkNode;
                var workItems = new List<Grid>() { current };
                int j = i - 1;
                while (j >= 0)
                {
                    var currentReads = current.Reads.AsValueEnumerable().Where(read => read is Grid producer && producer.Users.Count == 2).ToArray();
                    if (!(currentReads.Length == 1 && currentReads[0] == collects[j]))
                    {
                        break;
                    }

                    current = collects[j];
                    workItems.Add(current);
                    j--;
                }

                i = j;
                workLists.Insert(0, workItems);
            }
            else
            {
                i--;
            }
        }

        return workLists;
    }

    private sealed class AutoTileRewriter : ExprRewriter
    {
        public AutoTileRewriter(Grid rootGrid, CompileOptions compileOptions)
        {
            Root = rootGrid;
            CompileOptions = compileOptions;
            Wrapper = null!;
            PrimFunc = null!;
        }

        public Grid Root { get; }

        public CompileOptions CompileOptions { get; }

        public PrimFunctionWrapper Wrapper { get; set; }

        public TIR.PrimFunction PrimFunc { get; set; }

        protected override Expr RewriteLeafGrid(Grid grid)
        {
            if (grid == Root)
            {
                Expr call;
                (call, Wrapper, PrimFunc) = TreeTiler.Tile(grid, CompileOptions);
                return call;
            }

            return grid;
        }
    }
}
