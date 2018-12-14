using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Input;
using ReactiveUI;

namespace NnCase.Designer
{
    public abstract class LayoutItemBase : ReactiveObject, ILayoutItem
    {
        public Guid Id { get; } = Guid.NewGuid();

        public string ContentId { get; }

        public string Title { get; set; }

        public ICommand CloseCommand { get; set; }

        private bool _isSelected;
        public bool IsSelected
        {
            get => _isSelected;
            set => this.RaiseAndSetIfChanged(ref _isSelected, value);
        }

        public LayoutItemBase()
        {
            ContentId = Id.ToString();
        }
    }
}
