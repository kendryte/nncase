using System;
using System.Collections.Generic;
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

        string DefaultNamePrefix { get; }

        string Name { get; set; }

        double X { get; set; }

        double Y { get; set; }
    }

    public abstract class LayerViewModel<TModel> : ReactiveObject, ILayerViewModel
        where TModel : Layer
    {
        public abstract string DefaultNamePrefix { get; }

        public TModel Model { get; protected set; }

        private int[] _dimensions = new int[] { 1, 128, 128, 3 };

        public ReadOnlySpan<int> Dimensions
        {
            get => _dimensions;
            set
            {
                if (!value.SequenceEqual(_dimensions))
                {
                    _dimensions = value.ToArray();
                    this.RaisePropertyChanged();
                }
            }
        }

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
    }
}
