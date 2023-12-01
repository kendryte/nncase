﻿// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Diagnostics.CodeAnalysis;
using Avalonia;
using Avalonia.Media;
using Avalonia.Media.Fonts;
using Microsoft.Extensions.Hosting;

namespace Nncase.Studio;

public static class AvaloniaAppBuilderExtensions
{
    public static AppBuilder UseNncaseFontManager(this AppBuilder builder, Action<FontSettings>? configDelegate = default)
    {
        var setting = new FontSettings();
        configDelegate?.Invoke(setting);

        return builder.With(new FontManagerOptions
        {
            DefaultFamilyName = setting.DefaultFontFamily,
            FontFallbacks = new[]
            {
                new FontFallback
                {
                    FontFamily = new FontFamily(setting.DefaultFontFamily),
                },
            },
        }).ConfigureFonts(manager => manager.AddFontCollection(new EmbeddedFontCollection(setting.Key, setting.Source)));
    }

    public static AppBuilder ConfigureCompiler(this AppBuilder builder)
    {
        var host = Host.CreateDefaultBuilder()
            .ConfigureCompiler()
            .Build();
        CompilerServices.Configure(host.Services);
        return builder;
    }

    public static AppBuilder DisableDBus(this AppBuilder builder)
    {
        // https://github.com/AvaloniaUI/Avalonia/issues/9383#issuecomment-1378350456
        return builder.With(new X11PlatformOptions() { UseDBusFilePicker = false });
    }
}
