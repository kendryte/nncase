using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using NnCase.Designer.Services;
using ReactiveUI;

namespace NnCase.Designer.Modules.Shell.ViewModels
{
    public class ShellViewModel : ReactiveObject, IShell
    {
        public ObservableCollection<IDocument> Documents { get; } = new ObservableCollection<IDocument>();

        public ShellViewModel()
        {
        }

        private class TestDocument : Document
        {
        }
    }
}
