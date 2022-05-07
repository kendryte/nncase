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
/// Function id.
/// </summary>
/// <param name="Id">Id.</param>
/// <param name="ModuleId">Module id.</param>
public record FunctionId(int Id, int ModuleId);

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
    FunctionId GetFunctionId(Callable function);
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
    /// Gets attributes.
    /// </summary>
    int Attributes { get; }

    /// <summary>
    /// Gets alignment.
    /// </summary>
    int Alignment { get; }

    /// <summary>
    /// Gets size in file.
    /// </summary>
    int SizeInFile { get; }

    /// <summary>
    /// Gets size in memory.
    /// </summary>
    int SizeInMemory { get; }

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
    /// Gets parameter types.
    /// </summary>
    IReadOnlyList<IRType> ParameterTypes { get; }

    /// <summary>
    /// Gets return type.
    /// </summary>
    IRType ReturnType { get; }

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
    int Version { get; }

    /// <summary>
    /// Gets linked functions.
    /// </summary>
    IReadOnlyList<ILinkedFunction> Functions { get; }
}

public interface ILinkableModule
{
    ILinkedModule Link(ILinkContext linkContext);
}

public interface IModuleBuilder
{
    string ModuleKind { get; }

    ILinkableModule Build(IReadOnlyList<Callable> functions);
}
