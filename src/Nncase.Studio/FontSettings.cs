using System;

namespace Nncase.Studio;
public class FontSettings
{
    public string DefaultFontFamily = "fonts:CustomFontFamilies#Nunito";

    public Uri Key { get; set; } = new Uri("fonts:CustomFontFamilies", UriKind.Absolute);

    public Uri Source { get; set; } = new Uri("avares://Nncase.Studio/Assets/Fonts/Nunito", UriKind.Absolute);
}

