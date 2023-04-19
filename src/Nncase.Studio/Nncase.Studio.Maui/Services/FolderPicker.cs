using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Nncase.Studio.Services;

namespace Nncase.Studio.Maui.Services;

internal sealed class FolderPicker : IFolderPicker
{
    public async Task<string?> PickFolderAsync(string title, string? defaultPath = null)
    {
        var result = await CommunityToolkit.Maui.Storage.FolderPicker.PickAsync(default);
        return result.Folder?.Path;
    }
}
