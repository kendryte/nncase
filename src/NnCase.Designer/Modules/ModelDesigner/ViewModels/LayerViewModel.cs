using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using NnCase.Converter.Model;
using ReactiveUI;

namespace NnCase.Designer.Modules.ModelDesigner.ViewModels
{
    public interface ILayerViewModel : IReactiveObject
    {
        Layer Model { get; }

        string Name { get; set; }

        double X { get; set; }

        double Y { get; set; }

        bool IsSelected { get; set; }

        ObservableCollection<InputConnectorViewModel> InputConnectors { get; }

        ObservableCollection<OutputConnectorViewModel> OutputConnectors { get; }
    }

    public abstract class LayerViewModel<TModel> : ReactiveObject, ILayerViewModel
        where TModel : Layer
    {
        public TModel Model { get; protected set; }

        Layer ILayerViewModel.Model => Model;

        private double _x;
        public double X
        {
            get => _x;
            set => this.RaiseAndSetIfChanged(ref _x, value);
        }

        private double _y;
        public double Y
        {
            get => _y;
            set => this.RaiseAndSetIfChanged(ref _y, value);
        }

        private bool _isSelected;
        public bool IsSelected
        {
            get => _isSelected;
            set => this.RaiseAndSetIfChanged(ref _isSelected, value);
        }

        private string _name;
        public string Name
        {
            get => _name;
            set => this.RaiseAndSetIfChanged(ref _name, value);
        }

        public ObservableCollection<InputConnectorViewModel> InputConnectors { get; } = new ObservableCollection<InputConnectorViewModel>();

        public ObservableCollection<OutputConnectorViewModel> OutputConnectors { get; } = new ObservableCollection<OutputConnectorViewModel>();

        protected OutputConnectorViewModel AddOutput(string name, int[] dimensions)
        {
            var conn = new OutputConnectorViewModel(name, dimensions, this);
            OutputConnectors.Add(conn);
            return conn;
        }
    }
}
