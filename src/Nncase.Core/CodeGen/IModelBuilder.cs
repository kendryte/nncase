// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Nncase.IR;

namespace Nncase.CodeGen;

/// <summary>
/// Linked model.
/// </summary>
public interface ILinkedModel
{
    /// <summary>
    /// Gets entry function id.
    /// </summary>
    FunctionId? Entry { get; }

    /// <summary>
    /// Gets linked modules.
    /// </summary>
    IReadOnlyList<ILinkedModule> Modules { get; }

    /// <summary>
    /// Serialize model to stream.
    /// </summary>
    /// <param name="output">Stream to be written.</param>
    void Serialize(Stream output);
}

/// <summary>
/// Model builder.
/// </summary>
public interface IModelBuilder
{
    /// <summary>
    /// Build linked model.
    /// </summary>
    /// <param name="module">Module.</param>
    /// <returns>Linked model.</returns>
    ILinkedModel Build(IRModule module);
}
