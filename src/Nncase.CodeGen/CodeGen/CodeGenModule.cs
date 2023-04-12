// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using DryIoc;
using Nncase.CodeGen;
using Nncase.Diagnostics;
using Nncase.Hosting;

namespace Nncase.Diagnostics;

/// <summary>
/// CodeGen module.
/// </summary>
internal class CodeGenModule : IApplicationPart
{
    public void ConfigureServices(IRegistrator registrator)
    {
        registrator.Register<IModelBuilder, ModelBuilder>(reuse: Reuse.Scoped);
    }
}
