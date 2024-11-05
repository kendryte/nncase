// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;

namespace Nncase.Schedule.MonteCarloTreeSearch;

public abstract class Searcher<T>
    where T : class
{
    public Searcher(int searchTimes = 20)
    {
        SearchTimes = searchTimes;
    }

    public int SearchTimes { get; }

    public void Search(SearchNode<T> rootNode)
    {
        for (int i = 0; i < SearchTimes; i++)
        {
            if (!Selection(rootNode, out var node))
            {
                return;
            }

            var expanded = Expand(node);
            if (expanded is not null)
            {
                BackPropagate(expanded, Simulation(expanded));
            }
            else
            {
                BackPropagate(node, double.PositiveInfinity);
            }
        }
    }

    public abstract bool Selection(SearchNode<T> node, out SearchNode<T> selected);

    public abstract SearchNode<T>? Expand(SearchNode<T> node);

    public abstract double Simulation(SearchNode<T> node);

    public abstract void BackPropagate(SearchNode<T> node, double reward);
}
