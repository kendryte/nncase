// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.
#pragma warning disable SA1300

using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading.Tasks;
using WebKit;

namespace Nncase.Studio.Gtk.Blazor;

internal static class WebKit
{
    public const string FilePath = "webkit";

    public enum WebKitUserContentInjectedFrames
    {
        WEBKIT_USER_CONTENT_INJECT_ALL_FRAMES = 0,
        WEBKIT_USER_CONTENT_INJECT_TOP_FRAME = 1,
    }

    public enum WebKitUserScriptInjectionTime
    {
        WEBKIT_USER_SCRIPT_INJECT_AT_DOCUMENT_START = 0,
        WEBKIT_USER_SCRIPT_INJECT_AT_DOCUMENT_END = 1,
    }

    [DllImport(FilePath)]
    public static extern nint webkit_user_script_new(string source, WebKitUserContentInjectedFrames injected_frames, WebKitUserScriptInjectionTime injection_time, string? allow_list, string? block_list);

    [DllImport(FilePath)]
    public static extern void webkit_user_content_manager_add_script(nint manager, nint script);

    [DllImport(FilePath)]
    public static extern void webkit_user_script_unref(nint script);

    [DllImport(FilePath)]
    public static extern bool webkit_user_content_manager_register_script_message_handler(nint manager, string name);

    [DllImport(FilePath)]
    public static extern void webkit_web_view_run_javascript(nint web_view, string script, nint cancellable, nint callback, nint user_data);

    [DllImport(FilePath)]
    public static extern void webkit_javascript_result_unref(nint js_result);

    [DllImport(FilePath)]
    public static extern nint webkit_javascript_result_get_js_value(nint js_result);

    [DllImport(FilePath)]
    public static extern bool jsc_value_is_string(nint value);

    [DllImport(FilePath)]
    public static extern nint jsc_value_to_string(nint value);
}
