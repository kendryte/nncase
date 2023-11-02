using Avalonia;
using Avalonia.Controls;
using Avalonia.Markup.Xaml;
using Avalonia.ReactiveUI;
using CommunityToolkit.Mvvm.ComponentModel;
using Nncase.Studio.ViewModels;

namespace Nncase.Studio.Views;

public partial class PromptDialog : ReactiveWindow<PromptDialogViewModel>
{
    public PromptDialog()
    {
        InitializeComponent();
    }
}
