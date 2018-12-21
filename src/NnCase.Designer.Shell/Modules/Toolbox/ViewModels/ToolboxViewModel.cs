using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Data;
using NnCase.Designer.Services;
using NnCase.Designer.Toolbox;

namespace NnCase.Designer.Modules.Toolbox.ViewModels
{
    public class ToolboxViewModel : Tool, IToolbox
    {
        private readonly IToolboxService _toolboxService;

        public override PaneLocation PreferredLocation => PaneLocation.Left;

        public ObservableCollection<ToolboxItemViewModel> Items { get; }

        public ToolboxViewModel(IShell shell, IToolboxService toolboxService)
        {
            Title = "Toolbox";

            _toolboxService = toolboxService;
            Items = new ObservableCollection<ToolboxItemViewModel>();

            var groupedItems = CollectionViewSource.GetDefaultView(Items);
            groupedItems.GroupDescriptions.Add(new PropertyGroupDescription(nameof(ToolboxItem.Category)));

            shell.ActiveDocumentChanged += (s, e) => RefreshToolboxItems((IShell)s);
            RefreshToolboxItems(shell);
        }

        private void RefreshToolboxItems(IShell shell)
        {
            var activeDocument = shell.ActiveDocument;
            Items.Clear();

            if (activeDocument != null)
            {
                foreach (var item in _toolboxService.GetToolboxItems(activeDocument.GetType()))
                    Items.Add(new ToolboxItemViewModel(item));
            }
        }
    }
}
