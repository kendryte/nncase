// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using DryIoc;
using Nncase.Evaluator.Tensors;
using Nncase.Hosting;
using Nncase.IR.Shapes;

namespace Nncase.Evaluator.Shapes;

/// <summary>
/// Shapes module.
/// </summary>
internal class ShapesModule : IApplicationPart
{
    public void ConfigureServices(IRegistrator registrator)
    {
        registrator.RegisterManyInterface<AsTensorEvaluator>(reuse: Reuse.Singleton);
    }
}
