using CommunityToolkit.Mvvm.ComponentModel;

namespace Nncase.Studio.ViewModels;

public partial class PromptDialogViewModel : ViewModelBase
{
    [ObservableProperty] private string _dialogContent;
}
