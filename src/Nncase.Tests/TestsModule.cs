// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using DryIoc;
using Nncase.Hosting;

namespace Nncase.Tests;

/// <summary>
/// Test fixture module.
/// </summary>
internal sealed class TestsModule : IApplicationPart
{
    public void ConfigureServices(IRegistrator registrator)
    {
        registrator.RegisterManyInterface<TIRTest.ExtraWEvaluator>(reuse: Reuse.Singleton);
        registrator.RegisterManyInterface<TIRTest.LoadTEvaluator>(reuse: Reuse.Singleton);
        registrator.RegisterManyInterface<TIRTest.MeshNetEvaluator>(reuse: Reuse.Singleton);
    }
}
