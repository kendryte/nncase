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
        // window关联一个对应的ViewModel

        // foreach (string flag in Enum.GetNames(typeof(DumpFlags)))
        // {
        //     DumpFlagsComboBox.Items.Add(flag);
        // }
        //
        // foreach (string flag in Enum.GetNames(typeof(InputType)))
        // {
        //     InputType.Items.Add(flag);
        // }

        // InputLayout.Items.Add("NCHW");
        // InputLayout.Items.Add("NHWC");

        this.WhenActivated(action =>
        {
            action(ViewModel!.ShowPromptDialog.RegisterHandler(DoShowDialogAsync));
            action(ViewModel!.ShowFilePicker.RegisterHandler(OpenFileButton_Clicked));
        });
    }

    private async Task DoShowDialogAsync(InteractionContext<string, Unit> interaction)
    {
        var dialog = new PromptDialog();
        var viewModel = new PromptDialogViewModel();
        viewModel.DialogContent = interaction.Input;
        dialog.DataContext = viewModel;
        await dialog.ShowDialog(this);
    }

    public async Task OpenFileButton_Clicked(InteractionContext<FilePickerOpenOptions, List<string>> interaction)
    {
        // Get top level from the current control. Alternatively, you can use Window reference instead.
        var topLevel = TopLevel.GetTopLevel(this);


        // Start async operation to open the dialog.
        var files = await topLevel.StorageProvider.OpenFilePickerAsync(interaction.Input);

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
}
