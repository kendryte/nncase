using System;
using System.Collections.Generic;
using System.ComponentModel.DataAnnotations;
using System.Linq;
using System.Reflection;
using System.Text;
using System.Threading.Tasks;

namespace NnCase.Designer.Modules.Inspector.Inspectors
{
    public class EnumValueViewModel<TEnum>
    {
        public TEnum Value { get; set; }
        public string Text { get; set; }
    }

    public class EnumEditorViewModel<TEnum> : EditorBase<TEnum>, ILabelledInspector
    {
        private readonly List<EnumValueViewModel<TEnum>> _items;
        public IEnumerable<EnumValueViewModel<TEnum>> Items
        {
            get { return _items; }
        }

        public EnumEditorViewModel()
        {
            _items = Enum.GetValues(typeof(TEnum)).Cast<TEnum>().Select(x => new EnumValueViewModel<TEnum>
            {
                Value = x,
                Text = Enum.GetName(typeof(TEnum), x)
            }).ToList();
        }
    }

    public class EnumValueViewModel
    {
        public object Value { get; set; }
        public string Text { get; set; }
    }

    public class EnumEditorViewModel : EditorBase<Enum>, ILabelledInspector
    {
        private readonly List<EnumValueViewModel> _items;
        public IEnumerable<EnumValueViewModel> Items
        {
            get { return _items; }
        }

        public EnumEditorViewModel(Type enumType)
        {
            _items = (from f in enumType.GetFields(BindingFlags.Static | BindingFlags.Public)
                      let attr = f.GetCustomAttribute<DisplayAttribute>(false)
                      let value = f.GetValue(null)
                      select new EnumValueViewModel
                      {
                          Value = value,
                          Text = attr != null ? attr.GetName() : value.ToString()
                      }).ToList();
        }
    }
}
