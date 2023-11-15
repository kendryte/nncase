using CommunityToolkit.Mvvm.Input;

namespace Nncase.Studio.ViewModels;

public partial class StudioModeViewModel : ViewModelBase
{
    public StudioModeViewModel(ViewModelContext context)
    {
        Context = context;
    }

    [RelayCommand]
    void SwitchToImportView()
    {
        Context.SwitchToPage(typeof(ImportViewModel));
    }

    [RelayCommand]
    void SwitchToSimulateView()
    {
        Context.SwitchToPage(typeof(SimulateViewModel));
    }
}
