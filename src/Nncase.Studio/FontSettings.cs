// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;

namespace Nncase.Studio;

public class FontSettings
{
    public string DefaultFontFamily { get; set; } = "fonts:CustomFontFamilies#Nunito";

    public Uri Key { get; set; } = new Uri("fonts:CustomFontFamilies", UriKind.Absolute);

    public Uri Source { get; set; } = new Uri("avares://Nncase.Studio/Assets/Fonts/Nunito", UriKind.Absolute);
}
