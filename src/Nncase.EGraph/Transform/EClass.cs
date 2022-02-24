// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Nncase.Transform;

/// <summary>
/// EClass.
/// </summary>
public sealed class EClass
{
    /// <summary>
    /// Initializes a new instance of the <see cref="EClass"/> class.
    /// </summary>
    /// <param name="id">Id.</param>
    public EClass(int id)
    {
        Id = id;
    }

    /// <summary>
    /// Gets id.
    /// </summary>
    public int Id { get; }

    /// <summary>
    /// Gets or sets parent.
    /// </summary>
    public EClass? Parent { get; set; }

    /// <summary>
    /// NOTE the Used mean which Enode use this EClass. eg. z = x + y. the EClass's Used will add {(z, z's eclass id)}.
    /// <remark> It's Not mean this EClass's Nodes </remark>
    /// </summary>
    public readonly List<(ENode, EClass)> Used = new List<(ENode, EClass)>();

    /// <summary>
    /// Find root eclass.
    /// </summary>
    /// <returns>Root eclass.</returns>
    public EClass Find()
    {
        if (Parent is null)
        {
            return this;
        }

        Parent = Parent.Find();
        return Parent;
    }

    /// <inheritdoc/>
    public override string ToString() => $"{Id} -> {Parent?.Id}";
}
