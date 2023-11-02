using System;
using Avalonia;
using Avalonia.Controls;
using Avalonia.Data;
using Avalonia.Markup.Xaml;
using GiGraph.Dot.Entities.Html.Font.Styles;

namespace Nncase.Studio.Views
{
    public partial class EnumComboBox : ComboBox
    {
        public static readonly StyledProperty<Type> TypeNameProperty =
            AvaloniaProperty.Register<EnumComboBox, Type>(nameof(TypeName));

        protected override Type StyleKeyOverride => typeof(Avalonia.Controls.ComboBox);

        public Type TypeName
        {
            get { return GetValue(TypeNameProperty); }
            set
            {
                SetValue(TypeNameProperty, value);
                var names = Enum.GetNames(TypeName);
                foreach (string name in names)
                {
                    Items.Add(name);
                }
                SelectedIndex = 0;
            }
        }
    }
}

