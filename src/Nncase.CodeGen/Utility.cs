// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.IO;
using System.Runtime.InteropServices;

namespace Nncase.CodeGen;

public static partial class CodeGenUtil
{
    /// <summary>
    /// get temp file with extenstion.
    /// </summary>
    /// <param name="ext"> eg. "c".</param>
    /// <returns></returns>
    public static string GetTempFileName(string? ext = null)
    {
        ext ??= "tmp";
        if (!ext.StartsWith('.'))
        {
            ext = "." + ext;
        }

        return Path.GetTempPath() + Guid.NewGuid().ToString() + ext;
    }

    /// <summary>
    /// convert the c# struct to bytes.
    /// </summary>
    /// <param name="obj">the struct object instance.</param>
    /// <returns></returns>
    public static byte[] StructToBytes<T>(T obj)
    {
        int len = Marshal.SizeOf<T>();
        byte[] arr = new byte[len];
        IntPtr ptr = Marshal.AllocHGlobal(len);
        Marshal.StructureToPtr(obj, ptr, true);
        Marshal.Copy(ptr, arr, 0, len);
        Marshal.FreeHGlobal(ptr);
        return arr;
    }
}
