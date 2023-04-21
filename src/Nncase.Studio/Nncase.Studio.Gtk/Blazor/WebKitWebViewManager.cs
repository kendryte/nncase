// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading.Tasks;
using System.Web;
using Microsoft.AspNetCore.Components;
using Microsoft.AspNetCore.Components.Web;
using Microsoft.AspNetCore.Components.WebView;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.FileProviders;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Options;
using WebKit;

namespace Nncase.Studio.Gtk.Blazor;

public class WebKitWebViewManager : WebViewManager
{
    private delegate void void_nint_nint_nint(nint arg0, nint arg1, nint arg2);

    private static readonly string _appScheme = "app";
    private static readonly Uri _appBaseUri = new Uri($"{_appScheme}://localhost/");

    private ILogger<WebKitWebViewManager> _logger;
    private readonly WebView _webView;
    private readonly string _relativeHostPath;
    private readonly void_nint_nint_nint _handleWebMessageDelegate;

    public WebKitWebViewManager(WebView webView, IServiceProvider provider, Dispatcher dispatcher, IFileProvider fileProvider, JSComponentConfigurationStore jsComponents, IOptions<BlazorWebViewOptions> options)
        : base(provider, dispatcher, _appBaseUri, fileProvider, jsComponents, options.Value.RelativeHostPath)
    {
        _webView = webView;
        _relativeHostPath = options.Value.RelativeHostPath;
        _handleWebMessageDelegate = HandleWebMessage;
        _logger = provider.GetRequiredService<ILogger<WebKitWebViewManager>>();

        SetupSchemeHandler();
        SetupMessageChannel();
    }

    protected override void NavigateCore(Uri absoluteUri)
    {
        _logger.LogInformation($"Navigating to \"{absoluteUri}\"");

        _webView.LoadUri(absoluteUri.ToString());
    }

    protected override void SendMessage(string message)
    {
        _logger.LogDebug($"Dispatching `{message}`");

        var script = $"__dispatchMessageCallback(\"{HttpUtility.JavaScriptStringEncode(message)}\")";
        WebKit.webkit_web_view_run_javascript(_webView.Handle, script, nint.Zero, nint.Zero, nint.Zero);
    }

    private void SetupMessageChannel()
    {
        var scriptText = """
            window.__receiveMessageCallbacks = [];
            window.__dispatchMessageCallback = function(message) {
                window.__receiveMessageCallbacks.forEach(function(callback) { callback(message); });
            };
            window.external = {
                sendMessage: function(message) {
                    window.webkit.messageHandlers.webview.postMessage(message);
                },
                receiveMessage: function(callback) {
                    window.__receiveMessageCallbacks.push(callback);
                }
            };
            """;

        var script = WebKit.webkit_user_script_new(scriptText, WebKit.WebKitUserContentInjectedFrames.WEBKIT_USER_CONTENT_INJECT_ALL_FRAMES, WebKit.WebKitUserScriptInjectionTime.WEBKIT_USER_SCRIPT_INJECT_AT_DOCUMENT_START, null, null);
        WebKit.webkit_user_content_manager_add_script(_webView.UserContentManager.Handle, script);
        WebKit.webkit_user_script_unref(script);

        Gtk.g_signal_connect(_webView.UserContentManager.Handle, "script-message-received::webview",
            Marshal.GetFunctionPointerForDelegate(_handleWebMessageDelegate),
            nint.Zero);

        WebKit.webkit_user_content_manager_register_script_message_handler(_webView.UserContentManager.Handle, "webview");
    }

    private void SetupSchemeHandler()
    {
        // This is necessary to automatically serve the files in the `_framework` virtual folder.
        // Using `file://` will cause the webview to look for the `_framework` files on the file system,
        // and it won't find them.
        _webView.Context.RegisterUriScheme(_appScheme, HandleUriScheme);
    }

    private void HandleUriScheme(URISchemeRequest request)
    {
        if (request.Scheme != _appScheme)
        {
            throw new Exception($"Invalid scheme \"{request.Scheme}\"");
        }

        var uri = request.Uri;
        if (request.Path == "/")
        {
            uri += _relativeHostPath;
        }

        _logger.LogInformation($"Fetching \"{uri}\"");

        if (TryGetResponseContent(uri, false, out int statusCode, out string statusMessage, out Stream content, out IDictionary<string, string> headers))
        {
            using (var ms = new MemoryStream())
            {
                content.CopyTo(ms);

                var streamPtr = Gtk.g_memory_input_stream_new_from_data(ms.GetBuffer(), (uint)ms.Length, nint.Zero);
                var inputStream = new GLib.InputStream(streamPtr);
                request.Finish(inputStream, ms.Length, headers["Content-Type"]);
            }
        }
        else
        {
            throw new Exception($"Failed to serve \"{uri}\". {statusCode} - {statusMessage}");
        }
    }

    private void HandleWebMessage(nint contentManager, nint jsResult, nint arg)
    {
        var jsValue = WebKit.webkit_javascript_result_get_js_value(jsResult);

        if (WebKit.jsc_value_is_string(jsValue))
        {
            var p = WebKit.jsc_value_to_string(jsValue);
            var s = Marshal.PtrToStringAuto(p);
            if (s is not null)
            {
                 _logger.LogDebug($"Received message `{s}`");

                try
                {
                    MessageReceived(_appBaseUri, s);
                }
                finally
                {
                    Marshal.FreeHGlobal(p);
                }
            }
        }

        WebKit.webkit_javascript_result_unref(jsResult);
    }
}
