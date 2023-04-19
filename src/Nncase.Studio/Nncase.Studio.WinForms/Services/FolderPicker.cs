using Nncase.Studio.Services;
using Ookii.Dialogs.WinForms;

namespace Nncase.Studio.WinForms.Services;

internal sealed class FolderPicker : IFolderPicker
{
    public async Task<string> PickFolderAsync(string title)
    {
        using var folderPicker = new VistaFolderBrowserDialog
        {
            Description = title,
            UseDescriptionForTitle = true,
        };

        if (folderPicker.ShowDialog() == DialogResult.OK)
        {
            return folderPicker.SelectedPath;
        }

        return string.Empty;
    }
}
