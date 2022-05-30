// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Nncase.Runtime.Interop;

/// <summary>
/// Runtime object.
/// </summary>
public abstract class RTObject
{
    /// <summary>
    /// Initializes a new instance of the <see cref="RTObject"/> class.
    /// </summary>
    /// <param name="handle">Object handle.</param>
    internal RTObject(IntPtr handle)
    {
        Handle = handle;
    }

    /// <summary>
    /// Finalizes an instance of the <see cref="RTObject"/> class.
    /// </summary>
    ~RTObject()
    {
        Native.ObjectFree(Handle);
    }

    /// <summary>
    /// Gets object handle.
    /// </summary>
    public IntPtr Handle { get; }

    /// <inheritedoc/>
    public override bool Equals(object? obj)
    {
        return obj is RTObject @object &&
               EqualityComparer<IntPtr>.Default.Equals(Handle, @object.Handle);
    }

    /// <inheritedoc/>
    public override int GetHashCode()
    {
        return HashCode.Combine(Handle);
    }
}
