// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Logging;
using Nncase.IR;
using Nncase.IR.Affine;
using Nncase.Schedule;

namespace Nncase.Passes.Transforms;

public sealed class AutoTilePass : ModulePass
{
    private readonly Dictionary<AffineTilerMemo, GridSchedule> _memo = new();
    private readonly ILoggerFactory _loggerFactory;

    private readonly List<ExprPinner> _pinners = new();

    public AutoTilePass(CompileOptions compileOptions, ILoggerFactory loggerFactory)
    {
        CompileOptions = compileOptions;
        _loggerFactory = loggerFactory;
    }

    public CompileOptions CompileOptions { get; }

    protected override Task<IRModule> RunCoreAsync(IRModule input, RunPassContext context)
    {
        var funcs = input.Functions.Count;
        for (int i = 0; i < funcs; i++)
        {
            var rewriter = new AutoTileRewriter(input, CompileOptions.TargetCompileOptions, _memo, _pinners, _loggerFactory);
            input.Replace(i, (BaseFunction)rewriter.Rewrite(input.Functions[i]));
        }

        return Task.FromResult(input);
    }

    private sealed class AutoTileRewriter : ExprRewriter
    {
        private readonly IRModule _module;
        private readonly ITargetOptions _targetOptions;

        private readonly Dictionary<AffineTilerMemo, GridSchedule> _memo;

        private readonly List<ExprPinner> _pinners = new();
        private readonly ILoggerFactory _loggerFactory;

        public AutoTileRewriter(IRModule module, ITargetOptions targetOptions, Dictionary<AffineTilerMemo, GridSchedule> memo, List<ExprPinner> pinners, ILoggerFactory loggerFactory)
        {
            _module = module;
            _targetOptions = targetOptions;
            _memo = memo;
            _pinners = pinners;
            _loggerFactory = loggerFactory;
        }

        protected override Expr RewriteLeafGrid(Grid grid)
        {
            var scheduler = new AffineTiler(grid, _targetOptions, _loggerFactory);
            return scheduler.Tile(_module, _memo, _pinners);
        }
    }
}
