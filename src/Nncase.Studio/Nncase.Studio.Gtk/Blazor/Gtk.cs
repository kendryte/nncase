// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.
#pragma warning disable SA1300

using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading.Tasks;

namespace Nncase.Studio.Gtk.Blazor;

internal static class Gtk
{
    public const string FilePath = "gtk";

    public enum GtkWindowType : int
    {
        GTK_WINDOW_TOPLEVEL,
        GTK_WINDOW_WINDOW_POPUP,
    }

    public enum GConnectFlags : int
    {
        G_CONNECT_AFTER,
        G_CONNECT_SWAPPED,
    }

    [DllImport(FilePath)]
    public static extern ulong g_signal_connect_data(nint instance, string detailed_signal, nint c_handler, nint data, nint destroy_data, GConnectFlags connect_flags);

    public static ulong g_signal_connect(nint instance, string detailed_signal, nint c_handler, nint data)
    {
        return g_signal_connect_data(instance, detailed_signal, c_handler, data, nint.Zero, (GConnectFlags)0);
    }

    [DllImport(FilePath)]
    public static extern nint g_memory_input_stream_new_from_data(byte[] data, uint len, nint destroy);

    [DllImport(FilePath)]
    public static extern void g_free(IntPtr o);
}
