// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using Microsoft.Extensions.DependencyInjection;
using Nncase.Diagnostics;
using Nncase.IR;
using Nncase.TIR;

namespace Nncase.Transform;

/// <summary>
/// TIR Mutator Pass.
/// NOTE only apply on prim func
/// Because of we will mutate the expression multiple times, so use MutatorCreator create the new mutator.
/// </summary>
public class PrimFuncPass : Pass<PrimFunction>
{
    private readonly List<MutatorDescriptor> _mutatorDescriptors = new();

    /// <summary>
    /// Initializes a new instance of the <see cref="PrimFuncPass"/> class.
    /// </summary>
    public PrimFuncPass()
    {
    }

    /// <summary>
    /// Add mutator.
    /// </summary>
    /// <typeparam name="T">Mutator type.</typeparam>
    /// <param name="configureMutator">Configure mutator action.</param>
    /// <param name="arguments">Mutator's constructor arguments.</param>
    /// <returns>This primfunc pass.</returns>
    public PrimFuncPass Add<T>(Action<T>? configureMutator, params object[] arguments)
        where T : ExprMutator
    {
        _mutatorDescriptors.Add(new()
        {
            Factory = ActivatorUtilities.CreateFactory(typeof(T), arguments.Select(x => x.GetType()).ToArray()),
            Configure = configureMutator == null ? null : x => configureMutator?.Invoke((T)x),
            Arguments = arguments,
        });
        return this;
    }

    internal override string? GetDumpRelativePass(PrimFunction input) => input.Name;

    /// <inheritdoc/>
    protected override Task<PrimFunction> RunCoreAsync(PrimFunction input, RunPassContext context)
    {
        var post = input;
        int count = 0;
        bool isMutated = false;

        do
        {
            foreach (var descriptor in _mutatorDescriptors)
            {
                var mutator = descriptor.Activate(CompileSession);
                post = (PrimFunction)mutator.Visit(post);
                isMutated = mutator.IsMutated;

                if (isMutated)
                {
                    var typeInferSuccess = CompilerServices.InferenceType(post);
                    if (context.Dumpper.IsEnabled(DumpFlags.PassIR))
                    {
                        context.Dumpper.DumpIR(post, $"{count++}_{mutator.GetType().Name}");
                    }

                    Trace.Assert(typeInferSuccess);
                    break;
                }
            }

            if (!isMutated)
            {
                break;
            }
        }
        while (true);

        return Task.FromResult(post);
    }

    /// <inheritdoc/>
    protected override Task OnPassStartAsync(PrimFunction input, RunPassContext context) => Task.CompletedTask;

    /// <inheritdoc/>
    protected override Task OnPassEndAsync(PrimFunction post, RunPassContext context) => Task.CompletedTask;

    private struct MutatorDescriptor
    {
        public ObjectFactory Factory;
        public Action<ExprMutator>? Configure;
        public object[] Arguments;

        public ExprMutator Activate(CompileSession compileSession)
        {
            using var scope = new CompileSessionScope(compileSession);
            var mutator = (ExprMutator)Factory(compileSession.ServiceProvider, Arguments);
            Configure?.Invoke(mutator);
            return mutator;
        }
    }
}
