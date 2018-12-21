using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using NnCase.Designer.Modules.ModelDesigner.ViewModels.Layers;

namespace NnCase.Designer.Modules.ModelDesigner.ViewModels
{
    public class GraphViewModel : Document
    {
        public ObservableCollection<ILayerViewModel> Layers { get; } = new ObservableCollection<ILayerViewModel>();

        public GraphViewModel(string title)
        {
            Title = title;
        }
    }
}
