using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows;
using NnCase.Converter.Model;
using ReactiveUI;

namespace NnCase.Designer.Modules.ModelDesigner.ViewModels
{
    public class OutputConnectorViewModel : ReactiveObject
    {
        public string Name { get; }

        private readonly int[] _dimensions;

        public ReadOnlySpan<int> Dimensions => _dimensions;

        public string DimensionsText => $"{Dimensions[1]}x{Dimensions[2]}x{Dimensions[3]}";

        public LayerViewModel Owner { get; }

        private Point _position;
        public Point Position
        {
            get => _position;
            set
            {
                if (_position != value)
                {
                    _position = value;
                    this.RaisePropertyChanged();
                    PositionChanged?.Invoke(this, EventArgs.Empty);
                }
            }
        }

        public event EventHandler PositionChanged;

        public event EventHandler Updated;

        public bool IsInput => false;

        public OutputConnectorViewModel(string name, int[] dimensions, LayerViewModel owner)
        {
            Name = name;
            Owner = owner;
            _dimensions = dimensions;
        }

        public void SetDimension(Action<int[]> action)
        {
            action(_dimensions);
            this.RaisePropertyChanged(nameof(Dimensions));
            this.RaisePropertyChanged(nameof(DimensionsText));
            Updated?.Invoke(this, EventArgs.Empty);
        }
    }
}
