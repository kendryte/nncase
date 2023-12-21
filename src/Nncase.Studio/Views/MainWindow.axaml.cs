﻿// Copyright (c) Canaan Inc. All rights reserved.
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
            action(ViewModel!.ShowOpenFilePicker.RegisterHandler(OpenFileButtonClicked));
            action(ViewModel!.ShowSaveFilePicker.RegisterHandler(SaveFileButtonClicked));
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
            var path = files.Select(f => f.Path.LocalPath).ToList();
            interaction.SetOutput(path);
        }
        else
        {
            interaction.SetOutput(new List<string>());
        }
    }

    public async Task SaveFileButtonClicked(InteractionContext<FilePickerSaveOptions, string> interaction)
    {
        // Get top level from the current control. Alternatively, you can use Window reference instead.
        var topLevel = TopLevel.GetTopLevel(this);

        // Start async operation to open the dialog.
        var res = await topLevel!.StorageProvider.SaveFilePickerAsync(interaction.Input);
        if (res != null)
        {
            interaction.SetOutput(res.Path.LocalPath);
        }
        else
        {
            interaction.SetOutput(string.Empty);
        }
    }
}
