// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Reactive;
using System.Threading.Tasks;
using Avalonia;
using Avalonia.Controls;
using Avalonia.Interactivity;
using Avalonia.Media;
using Avalonia.Platform.Storage;
using Avalonia.ReactiveUI;
using Nncase.Diagnostics;
using Nncase.Studio.ViewModels;
using ReactiveUI;

namespace Nncase.Studio.Views;

public partial class MainWindow : ReactiveWindow<MainWindowViewModel>
{
    public MainWindow()
    {
        InitializeComponent();
        this.WhenActivated(action =>
        {
            action(ViewModel!.ShowPromptDialog.RegisterHandler(DoShowDialogAsync));
            action(ViewModel!.ShowFilePicker.RegisterHandler(OpenFileButtonClicked));
            action(ViewModel!.ShowFolderPicker.RegisterHandler(OpenFolderButtonClicked));
        });
    }

    public async Task OpenFolderButtonClicked(InteractionContext<FolderPickerOpenOptions, string> interaction)
    {
        // Get top level from the current control. Alternatively, you can use Window reference instead.
        var topLevel = TopLevel.GetTopLevel(this);

        // Start async operation to open the dialog.
        var folder = await topLevel!.StorageProvider.OpenFolderPickerAsync(interaction.Input);
        if (folder.Count == 0)
        {
            interaction.SetOutput(string.Empty);
            return;
        }

        interaction.SetOutput(folder[0].Path.LocalPath);
    }

    public async Task OpenFileButtonClicked(InteractionContext<FilePickerOpenOptions, List<string>> interaction)
    {
        // Get top level from the current control. Alternatively, you can use Window reference instead.
        var topLevel = TopLevel.GetTopLevel(this);

        // Start async operation to open the dialog.
        var files = await topLevel!.StorageProvider.OpenFilePickerAsync(interaction.Input);

        if (files.Count >= 1)
        {
            Console.WriteLine(files[0].Path.LocalPath);
            var path = files.Select(f => f.Path.LocalPath).ToList();
            interaction.SetOutput(path);
        }
        else
        {
            interaction.SetOutput(new List<string>());
        }
    }

    private async Task DoShowDialogAsync(InteractionContext<(string Message, PromptDialogLevel Level), Unit> interaction)
    {
        var dialog = new PromptDialog();
        var (content, level) = interaction.Input;
        var viewModel = new PromptDialogViewModel(content, level);
        interaction.SetOutput(default);
        dialog.DataContext = viewModel;
        await dialog.ShowDialog(this);
    }
}
