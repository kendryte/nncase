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
        public IMenu MainMenu { get; }

        public ObservableCollection<IDocument> Documents { get; } = new ObservableCollection<IDocument>();

        private ILayoutItem _activeLayoutItem;

        public ILayoutItem ActiveLayoutItem
        {
            get => _activeLayoutItem;
            set
            {
                if (_activeLayoutItem != value)
                {
                    _activeLayoutItem = value;
                    if (_activeLayoutItem is IDocument document)
                        OpenDocument(document);
                    this.RaisePropertyChanged();
                }
            }
        }

        public ShellViewModel(IMenu mainMenu)
        {
            MainMenu = mainMenu;
        }

        public void OpenDocument(IDocument document)
        {
            if (ActiveLayoutItem != document)
            {
                if (!Documents.Contains(document))
                    Documents.Add(document);
                ActiveLayoutItem = document;
            }
        }
    }
}
