using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using NnCase.Designer.Toolbox;
using ReactiveUI;

namespace NnCase.Designer.Modules.Toolbox.ViewModels
{
    public class ToolboxItemViewModel : ReactiveObject
    {
        public ToolboxItem Model { get; }

        public string Text => Model.Text;

        public string Category => Model.Category;

        public ToolboxItemViewModel(ToolboxItem model)
        {
            Model = model;
        }
    }
}
