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
    public abstract class LayerViewModel: ReactiveObject
    {
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

        protected InputConnectorViewModel AddInput(string name)
        {
            var conn = new InputConnectorViewModel(name, this);
            InputConnectors.Add(conn);
            return conn;
        }

        protected OutputConnectorViewModel AddOutput(string name, int[] dimensions)
        {
            var conn = new OutputConnectorViewModel(name, dimensions, this);
            OutputConnectors.Add(conn);
            return conn;
        }

        public void BuildModel(BuildGraphContext context)
        {
            if (context.Layers.ContainsKey(this)) return;
            
            foreach (var input in InputConnectors)
            {
                if (input.Connection == null)
                    throw new InvalidOperationException("Input not satisfied.");

                input.Connection.From.Owner.BuildModel(context);
            }

            BuildModelCore(context);
        }

        public void BuildConnections(BuildGraphContext context)
        {
            foreach (var input in InputConnectors)
            {
                input.Connection.From.Owner.BuildConnections(context);
                input.Build(context);
            }
        }

        protected abstract void BuildModelCore(BuildGraphContext context);
    }
}
