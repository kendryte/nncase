using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using NnCase.Designer.Services;
using ReactiveUI;

namespace NnCase.Designer.Modules.MainWindow.ViewModels
{
    public class MainWindowViewModel : ReactiveObject
    {
        public IShell Shell { get; }

        public MainWindowViewModel(IShell shell)
        {
            Shell = shell;
        }
    }
}
