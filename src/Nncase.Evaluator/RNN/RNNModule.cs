// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using DryIoc;
using Nncase.Evaluator.NN;
using Nncase.Hosting;

namespace Nncase.Evaluator.RNN;

/// <summary>
/// RNN module.
/// </summary>
internal class RNNModule : IApplicationPart
{
    public void ConfigureServices(IRegistrator registrator)
    {
        registrator.RegisterManyInterface<LSTMEvaluator>(reuse: Reuse.Singleton);
    }
}
