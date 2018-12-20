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
        public override PaneLocation PreferredLocation => PaneLocation.Left;

        public ObservableCollection<ToolboxItem> Items { get; }

        public ToolboxViewModel(IShell shell, IToolboxService toolboxService)
        {
            Items = new ObservableCollection<ToolboxItem>();

            var groupedItems = CollectionViewSource.GetDefaultView(Items);
            groupedItems.GroupDescriptions.Add(new PropertyGroupDescription(nameof(ToolboxItem.Category)));
        }
    }
}
