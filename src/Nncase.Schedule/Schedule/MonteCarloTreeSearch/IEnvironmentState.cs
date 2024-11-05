// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;

namespace Nncase.Schedule.MonteCarloTreeSearch;

public interface IEnvironmentState<TAction>
    where TAction : class
{
    int LegalActions();

    TAction GetNextAction(int index);

    IEnvironmentState<TAction>? PerformAction(TAction action);

    double RollOut();
}
