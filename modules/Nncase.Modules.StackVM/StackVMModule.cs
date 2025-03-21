// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using DryIoc;
using Nncase.CodeGen;
using Nncase.CodeGen.StackVM;
using Nncase.Hosting;
using Nncase.Targets;

namespace Nncase;

/// <summary>
/// StackVM module.
/// </summary>
internal class StackVMModule : IApplicationPart
{
    public void ConfigureServices(IRegistrator registrator)
    {
        registrator.Register<IStackVMModuleBuilder, StackVMModuleBuilder>(reuse: Reuse.Singleton);
    }
}
