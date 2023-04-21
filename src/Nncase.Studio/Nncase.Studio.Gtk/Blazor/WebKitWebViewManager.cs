// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Web;
using Microsoft.AspNetCore.Components;
using Microsoft.AspNetCore.Components.Web;
using Microsoft.AspNetCore.Components.WebView;
using Microsoft.Extensions.FileProviders;
using Microsoft.Extensions.Options;
using WebKit;

namespace Nncase.Studio.Gtk.Blazor;

public class WebKitWebViewManager : WebViewManager
{
    public static readonly string BlazorAppScheme = "app";
    public static readonly Uri AppBaseUri = new Uri($"{BlazorAppScheme}://localhost/");

    private readonly WebView _webView;

    public WebKitWebViewManager(WebView webView, IServiceProvider provider, Dispatcher dispatcher, IFileProvider fileProvider, JSComponentConfigurationStore jsComponents, IOptions<BlazorWebViewOptions> options)
        : base(provider, dispatcher, AppBaseUri, fileProvider, jsComponents, options.Value.RelativeHostPath)
    {
        _webView = webView;

        SetupMessageChannel();
    }

    protected override void NavigateCore(Uri absoluteUri)
    {
        _webView.LoadUri(absoluteUri.ToString());
    }

    protected override void SendMessage(string message)
    {
        var script = $"__dispatchMessageCallback(\"{HttpUtility.JavaScriptStringEncode(message)}\")";
        _webView.RunJavascript(script);
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
        var script = WebKit.CreateUserScript(scriptText, WebKit.WebKitUserContentInjectedFrames.WEBKIT_USER_CONTENT_INJECT_ALL_FRAMES, WebKit.WebKitUserScriptInjectionTime.WEBKIT_USER_SCRIPT_INJECT_AT_DOCUMENT_START);
        _webView.UserContentManager.AddScript(script);
    }
}
