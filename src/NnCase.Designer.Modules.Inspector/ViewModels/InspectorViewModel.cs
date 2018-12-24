using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using NnCase.Designer.Toolbox;
using ReactiveUI;

namespace NnCase.Designer.Modules.Inspector.ViewModels
{
    public class InspectorViewModel : Tool, IInspectorTool
    {
        private IInspectableObject _selectedObject;

        public override PaneLocation PreferredLocation => PaneLocation.Right;

        public IInspectableObject SelectedObject
        {
            get => _selectedObject;
            set
            {
                if (_selectedObject != value)
                {
                    _selectedObject = value;
                    this.RaisePropertyChanged();
                }
            }
        }

        public InspectorViewModel()
        {
            Title = "Inspector";
        }
    }
}
