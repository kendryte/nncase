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
/// Link context.
/// </summary>
public interface ILinkContext
{
    /// <summary>
    /// Gets function id.
    /// </summary>
    /// <param name="function">Function.</param>
    /// <returns>Function id.</returns>
    FunctionId GetFunctionId(BaseFunction function);
}

/// <summary>
/// Linked section.
/// </summary>
public interface ILinkedSection
{
    /// <summary>
    /// Gets section name.
    /// </summary>
    string Name { get; }

    /// <summary>
    /// Gets flags.
    /// </summary>
    uint Flags { get; }

    /// <summary>
    /// Gets alignment.
    /// </summary>
    uint Alignment { get; }

    /// <summary>
    /// Gets size in file.
    /// </summary>
    uint SizeInFile { get; }

    /// <summary>
    /// Gets size in memory.
    /// </summary>
    uint SizeInMemory { get; }

    /// <summary>
    /// Serialize payload.
    /// </summary>
    /// <param name="output">Output stream.</param>
    void Serialize(Stream output);
}

/// <summary>
/// Linked function.
/// </summary>
public interface ILinkedFunction
{
    /// <summary>
    /// Gets id.
    /// </summary>
    uint Id { get; }

    /// <summary>
    /// Gets parameter types.
    /// </summary>
    IReadOnlyList<IRType> ParameterTypes { get; }

    /// <summary>
    /// Gets return type.
    /// </summary>
    IRType ReturnType { get; }

    /// <summary>
    /// Gets text begin.
    /// </summary>
    uint TextBegin { get; }

    /// <summary>
    /// Gets text length.
    /// </summary>
    uint TextLength { get; }

    /// <summary>
    /// Gets sections.
    /// </summary>
    IReadOnlyList<ILinkedSection> Sections { get; }
}

/// <summary>
/// Linked module.
/// </summary>
public interface ILinkedModule
{
    /// <summary>
    /// Gets module kind.
    /// </summary>
    string ModuleKind { get; }

    /// <summary>
    /// Gets module version.
    /// </summary>
    uint Version { get; }

    /// <summary>
    /// Gets linked functions.
    /// </summary>
    IReadOnlyList<ILinkedFunction> Functions { get; }

    /// <summary>
    /// Gets sections.
    /// </summary>
    IReadOnlyList<ILinkedSection> Sections { get; }
}

/// <summary>
/// Linkable module.
/// </summary>
public interface ILinkableModule
{
    /// <summary>
    /// Link module.
    /// </summary>
    /// <param name="linkContext">Link context.</param>
    /// <returns>Linked module.</returns>
    ILinkedModule Link(ILinkContext linkContext);
}

/// <summary>
/// Module builder.
/// </summary>
public interface IModuleBuilder
{
    /// <summary>
    /// Gets module kind.
    /// </summary>
    string ModuleKind { get; }

    /// <summary>
    /// Build linkable module.
    /// </summary>
    /// <param name="functions">Source functions.</param>
    /// <param name="orderedNames">Ordered Names.</param>
    /// <returns>Compiled linkable module.</returns>
    ILinkableModule Build(IReadOnlyList<BaseFunction> functions, List<string> orderedNames);
}

/// <summary>
/// Function id.
/// </summary>
/// <param name="Id">Id.</param>
/// <param name="ModuleId">Module id.</param>
public record FunctionId(uint Id, uint ModuleId);
