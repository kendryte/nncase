// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Nncase.IR;

namespace Nncase.Passes;

/// <summary>
/// EGraph interface.
/// </summary>
public interface IEGraph
{
    /// <summary>
    /// Gets version.
    /// </summary>
    int Version { get; }

    /// <summary>
    /// Gets root class.
    /// </summary>
    EClass? Root { get; }

    /// <summary>
    /// Gets eclasses.
    /// </summary>
    IEnumerable<EClass> Classes { get; }

    /// <summary>
    /// Gets nodes.
    /// </summary>
    IEnumerable<ENode> Nodes { get; }

    /// <summary>
    /// Add expr, get the eclass id.
    /// </summary>
    /// <param name="expr">Expression.</param>
    /// <returns>Eclass of this node.</returns>
    EClass Add(Expr expr);

    /// <summary>
    /// Find eclass of enode.
    /// </summary>
    /// <param name="node">ENode.</param>
    /// <returns>EClass.</returns>
    EClass Find(ENode node);

    /// <summary>
    /// Union two equal Eclass.
    /// </summary>
    /// <param name="classA">class a.</param>
    /// <param name="classB">class b.</param>
    /// <returns>If version changed.</returns>
    bool Union(EClass classA, EClass classB);

    /// <summary>
    /// After merge, we use rebuild get new dep information.
    /// </summary>
    void Rebuild();
}
