// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Nncase.Studio.Gtk.Blazor;

public record BlazorWebViewOptions
{
    public string HostPath { get; set; } = Path.Combine("wwwroot", "index.html");

    public string ContentRoot { get => Path.GetDirectoryName(Path.GetFullPath(HostPath))!; }

    public string RelativeHostPath { get => Path.GetRelativePath(ContentRoot, HostPath); }
}
