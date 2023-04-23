// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Gtk;
using Nncase.Studio.Services;

namespace Nncase.Studio.Photino.Services;

internal sealed class FolderPicker : IFolderPicker
{
    public Task<string?> PickFolderAsync(string title, string? defaultPath = null)
    {
        using var fileChooser = new FileChooserDialog(
            title,
            null!,
            FileChooserAction.SelectFolder,
            "Cancel",
            ResponseType.Cancel,
            "Open",
            ResponseType.Accept);
        if (fileChooser.Run() == (int)ResponseType.Accept)
        {
            return Task.FromResult<string?>(fileChooser.Filename);
        }

        return Task.FromResult<string?>(null);
    }
}
