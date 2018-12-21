using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using NnCase.Designer.Services;
using ReactiveUI;
using Splat;

namespace NnCase.Designer.Modules.Shell.ViewModels
{
    public class ShellViewModel : ReactiveObject, IShell
    {
        public IMenu MainMenu { get; }

        public ObservableCollection<IDocument> Documents { get; } = new ObservableCollection<IDocument>();

        private ILayoutItem _activeLayoutItem;
        private IDocument _activeDocument;

        public event EventHandler ActiveDocumentChanged;

        public ILayoutItem ActiveLayoutItem
        {
            get => _activeLayoutItem;
            set
            {
                if (_activeLayoutItem != value)
                {
                    _activeLayoutItem = value;
                    if (_activeLayoutItem is IDocument document)
                    {
                        OpenDocument(document);
                        ActiveDocument = document;
                    }

                    this.RaisePropertyChanged();
                }
            }
        }

        public IDocument ActiveDocument
        {
            get => _activeDocument;
            private set
            {
                if (_activeDocument != value)
                {
                    _activeDocument = value;
                    this.RaisePropertyChanged();
                    ActiveDocumentChanged?.Invoke(this, EventArgs.Empty);
                }
            }
        }

        public ObservableCollection<ITool> Tools { get; } = new ObservableCollection<ITool>();

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

        public void OpenTool<TTool>() where TTool : ITool
        {
            OpenTool(Locator.Current.GetService<TTool>());
        }

        private void OpenTool(ITool tool)
        {
            if (ActiveLayoutItem != tool)
            {
                if (!Tools.Contains(tool))
                    Tools.Add(tool);
                ActiveLayoutItem = tool;
            }
        }
    }
}
