using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Microsoft.AspNetCore.Components;
using Nncase.Studio.Services;

namespace Nncase.Studio.Pages;

public partial class Welcome
{
    [Inject]
    public NavigationManager NavigationManager { get; set; } = default!;

    [Inject]
    public IFolderPicker FolderPicker { get; set; } = default!;

    private async void OpenWorkspace()
    {
        var folder = await FolderPicker.PickFolderAsync("打开工作区");
        if (!string.IsNullOrEmpty(folder))
        {
            NavigationManager.NavigateTo($"workspace/{folder}");
        }
    }
}
