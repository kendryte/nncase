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
    public AutoTilePass(string moduleKind, CompileOptions compileOptions)
    {
        ModuleKind = moduleKind;
        CompileOptions = compileOptions;
        WorkItem = 0;
    }

    public string ModuleKind { get; }

    public CompileOptions CompileOptions { get; }

    public int WorkItem { get; set; }

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
                var rewriter = new AutoTileRewriter(rootGrid, ModuleKind, WorkItem++, CompileOptions);
                post = (BaseFunction)rewriter.Rewrite(post);
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
                    var currentReads = current.Reads.AsValueEnumerable().Where(read => read is Grid producer && producer.Users.Count() == 2).ToArray();
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
        private readonly string _moduleKind;
        private readonly int _workItem;

        public AutoTileRewriter(Grid rootGrid, string moduleKind, int workItem, CompileOptions compileOptions)
        {
            Root = rootGrid;
            _moduleKind = moduleKind;
            _workItem = workItem;
            CompileOptions = compileOptions;
        }

        public Grid Root { get; }

        public CompileOptions CompileOptions { get; }

        protected override Expr RewriteLeafGrid(Grid grid)
        {
            if (grid == Root)
            {
                return TreeTiler.Tile(grid, _moduleKind, _workItem, CompileOptions.TargetOptions);
            }

            return grid;
        }
    }
}
