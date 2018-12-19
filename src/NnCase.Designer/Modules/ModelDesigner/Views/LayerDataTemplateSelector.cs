using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Controls;

namespace NnCase.Designer.Modules.ModelDesigner.Views
{
    public class TypedDataTemplateSelector : DataTemplateSelector
    {
        public Dictionary<Type, DataTemplate> Templates { get; } = new Dictionary<Type, DataTemplate>();

        public DataTemplate Default { get; set; }

        public override DataTemplate SelectTemplate(object item, DependencyObject container)
        {
            if (item != null)
            {
                if (Templates.TryGetValue(item.GetType(), out var template))
                    return template;
                return Default;
            }

            return base.SelectTemplate(item, container);
        }
    }
}
