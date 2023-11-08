// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

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

        public Type TypeName
        {
            get
            {
                return GetValue(TypeNameProperty);
            }

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

        protected override Type StyleKeyOverride => typeof(Avalonia.Controls.ComboBox);
    }
}
