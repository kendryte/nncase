// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System.Runtime.InteropServices;

namespace Nncase.Runtime.Interop;

/// <summary>
/// Runtime object.
/// </summary>
public abstract class RTObject : SafeHandle
{
    /// <summary>
    /// Initializes a new instance of the <see cref="RTObject"/> class.
    /// </summary>
    /// <param name="handle">Object handle.</param>
    internal RTObject(IntPtr handle) : base(handle, true)
    {
    }

    /// <inheritdoc/>
    protected override bool ReleaseHandle()
    {
        return Native.ObjectFree(handle).IsSuccess;
    }

    /// <inheritedoc/>
    public override bool Equals(object? obj)
    {
        return obj is RTObject @object && EqualityComparer<IntPtr>.Default.Equals(handle, @object.handle);
    }

    /// <inheritedoc/>
    public override int GetHashCode()
    {
        return HashCode.Combine(handle);
    }
}
