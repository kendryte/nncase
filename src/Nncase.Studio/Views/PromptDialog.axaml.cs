// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Avalonia;
using Avalonia.Controls;
using Avalonia.Markup.Xaml;
using Avalonia.ReactiveUI;
using CommunityToolkit.Mvvm.ComponentModel;
using Nncase.Studio.ViewModels;
using ReactiveUI;

namespace Nncase.Studio.Views;

public partial class PromptDialog : ReactiveWindow<PromptDialogViewModel>
{
    public PromptDialog()
    {
        InitializeComponent();
    }
}
