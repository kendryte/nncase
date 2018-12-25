using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using NnCase.Designer.Modules.Inspector.Inspectors;

namespace NnCase.Designer.Modules.Inspector.Conventions
{
    public static class DefaultPropertyInspectors
    {
        private static readonly List<PropertyEditorBuilder> _inspectorBuilders;

        public static List<PropertyEditorBuilder> InspectorBuilders
        {
            get { return _inspectorBuilders; }
        }

        static DefaultPropertyInspectors()
        {
            _inspectorBuilders = new List<PropertyEditorBuilder>
            {
                //new RangePropertyEditorBuilder(),
                new EnumPropertyEditorBuilder(),
                //
                //new StandardPropertyEditorBuilder<bool, CheckBoxEditorViewModel>(),
                //new StandardPropertyEditorBuilder<Color, ColorEditorViewModel>(),
                //new StandardPropertyEditorBuilder<double, TextBoxEditorViewModel<double>>(),
                //new StandardPropertyEditorBuilder<float, TextBoxEditorViewModel<float>>(),
                new StandardPropertyEditorBuilder<int, TextBoxEditorViewModel<int>>(),
                //new StandardPropertyEditorBuilder<double?, TextBoxEditorViewModel<double?>>(),
                //new StandardPropertyEditorBuilder<float?, TextBoxEditorViewModel<float?>>(),
                new StandardPropertyEditorBuilder<int?, TextBoxEditorViewModel<int?>>(),
                //new StandardPropertyEditorBuilder<Point3D, Point3DEditorViewModel>(),
                new StandardPropertyEditorBuilder<string, TextBoxEditorViewModel<string>>(),

                //new StandardPropertyEditorBuilder<BitmapSource, BitmapSourceEditorViewModel>()
            };
        }

        public static IEditor CreateEditor(PropertyDescriptor propertyDescriptor)
        {
            foreach (var inspectorBuilder in _inspectorBuilders)
            {
                if (inspectorBuilder.IsApplicable(propertyDescriptor))
                    return inspectorBuilder.BuildEditor(propertyDescriptor);
            }
            return null;
        }
    }
}
