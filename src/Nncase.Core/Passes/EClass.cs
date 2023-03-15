// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Nncase.Passes;

/// <summary>
/// EClass.
/// </summary>
public sealed class EClass
{
    private readonly List<ENode> _nodes = new();
    private List<ENode>? _usedBy = new();

    /// <summary>
    /// Initializes a new instance of the <see cref="EClass"/> class.
    /// </summary>
    /// <param name="id">Id.</param>
    public EClass(int id)
    {
        Id = id;
    }

    /// <summary>
    /// Gets the eclass's ir type.
    /// </summary>
    public IR.IRType CheckedType { get; private set; } = IR.AnyType.Default;

    /// <summary>
    /// Gets id.
    /// </summary>
    public int Id { get; }

    /// <summary>
    /// Gets or sets parent.
    /// </summary>
    public EClass? Parent { get; set; }

    /// <summary>
    /// Gets the used by mean which Enode use this EClass. eg. z = x + y. the EClass's Used will add {(z, z's eclass id)}.
    /// <remark> It's Not mean this EClass's Nodes </remark>
    /// </summary>
    public IReadOnlyList<ENode> UsedBy => _usedBy ?? throw new InvalidOperationException("This class has been merged.");

    /// <summary>
    /// Gets nodes.
    /// </summary>
    public IReadOnlyList<ENode> Nodes => _nodes;

    /// <summary>
    /// Set the new checked type and we need update the all inner enode expr with new type.
    /// </summary>
    public void SetCheckedType(IR.IRType type)
    {
        CheckedType = type;
    }

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

    /// <summary>
    /// Add enode.
    /// </summary>
    /// <param name="enode">ENode.</param>
    public void AddNode(ENode enode)
    {
        _nodes.Add(enode);
    }

    /// <summary>
    /// Add enode.
    /// </summary>
    /// <param name="enodes">ENodes.</param>
    public void AddNodes(IEnumerable<ENode> enodes)
    {
        _nodes.AddRange(enodes);
    }

    /// <summary>
    /// Remove enode.
    /// </summary>
    /// <param name="enode">ENode.</param>
    public void RemoveNode(ENode enode)
    {
        _nodes.Remove(enode);
    }

    /// <summary>
    /// Add used by enode.
    /// </summary>
    /// <param name="enode">ENode.</param>
    public void AddUsedBy(ENode enode)
    {
        if (_usedBy == null)
        {
            throw new InvalidOperationException("This class has been merged.");
        }

        _usedBy.Add(enode);
    }

    /// <summary>
    /// Kill this class.
    /// </summary>
    public void Kill()
    {
        _nodes.Clear();
        _usedBy = null;
    }

    /// <inheritdoc/>
    public override string ToString() => $"{Id} -> {Parent?.Id}";
}
