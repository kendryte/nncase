// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;

namespace Nncase.Schedule.MonteCarloTreeSearch;

public abstract class SearchNode<T>
    where T : class
{
    public SearchNode(IEnvironmentState<T> state)
    {
        Parent = null;
        Children = new List<SearchNode<T>>();
        VisitTimes = 0;
        QualityValue = 0.0;
        State = state;
    }

    public SearchNode(SearchNode<T> parent, IEnvironmentState<T> state)
    {
        Parent = parent;
        Children = new List<SearchNode<T>>();
        VisitTimes = 0;
        QualityValue = 0.0;
        State = state;
        Parent.Children.Add(this);
    }

    public SearchNode<T>? Parent { get; }

    public List<SearchNode<T>> Children { get; }

    public int VisitTimes { get; set; }

    public double QualityValue { get; set; }

    public IEnvironmentState<T> State { get; }

    public bool IsRootNode => Parent is null;

    public abstract void Update(double reward);
}
