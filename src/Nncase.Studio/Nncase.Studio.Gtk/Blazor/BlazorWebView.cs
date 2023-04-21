// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Collections.Specialized;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Microsoft.AspNetCore.Components;
using Microsoft.AspNetCore.Components.WebView;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.FileProviders;
using Microsoft.Extensions.Options;
using WebKit;

namespace Nncase.Studio.Gtk.Blazor;

internal sealed class BlazorWebView : WebView
{
    private readonly WebKitWebViewManager _webViewManager;

    public BlazorWebView(IServiceProvider serviceProvider)
    {
        Dispatcher = serviceProvider.GetRequiredService<Dispatcher>();

        var options = serviceProvider.GetRequiredService<IOptions<BlazorWebViewOptions>>();
        var fileProvider = new PhysicalFileProvider(options.Value.ContentRoot);
        _webViewManager = new WebKitWebViewManager(this, serviceProvider, serviceProvider.GetRequiredService<Dispatcher>(), fileProvider, RootComponents.JSComponents, options);

        foreach (var rootComponent in RootComponents)
        {
            // Since the page isn't loaded yet, this will always complete synchronously
            _ = rootComponent.AddToWebViewManagerAsync(_webViewManager);
        }

        RootComponents.CollectionChanged += HandleRootComponentsCollectionChanged;
        _webViewManager.Navigate("/");
    }

    public Dispatcher Dispatcher { get; }

    public RootComponentsCollection RootComponents { get; } = new();

    protected override void Dispose(bool disposing)
    {
        if (disposing)
        {
            // Dispose this component's contents and block on completion so that user-written disposal logic and
            // Blazor disposal logic will complete first. Then call base.Dispose(), which will dispose the WebKit
            // control. This order is critical because once the WebKit is disposed it will prevent and Blazor
            // code from working because it requires the WebView to exist.
            _webViewManager
                .DisposeAsync()
                .AsTask()
                .GetAwaiter()
                .GetResult();
        }

        base.Dispose(disposing);
    }

    private void HandleRootComponentsCollectionChanged(object? sender, NotifyCollectionChangedEventArgs eventArgs)
    {
        // Dispatch because this is going to be async, and we want to catch any errors
        _ = Dispatcher.InvokeAsync(async () =>
        {
            var newItems = eventArgs.NewItems!.Cast<RootComponent>();
            var oldItems = eventArgs.OldItems!.Cast<RootComponent>();

            foreach (var item in newItems.Except(oldItems))
            {
                await item.AddToWebViewManagerAsync(_webViewManager);
            }

            foreach (var item in oldItems.Except(newItems))
            {
                await item.RemoveFromWebViewManagerAsync(_webViewManager);
            }
        });
    }
}
