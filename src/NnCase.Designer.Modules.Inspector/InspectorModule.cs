using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Autofac;
using ReactiveUI;

namespace NnCase.Designer.Modules.Inspector
{
    public class InspectorModule : Module
    {
        protected override void Load(ContainerBuilder builder)
        {
            builder.RegisterType<ViewModels.InspectorViewModel>()
                .AsSelf().As<IInspectorTool>().SingleInstance();

            builder.RegisterType<Views.InspectorView>();
            builder.RegisterType<Inspectors.CollapsibleGroupView>();
            builder.RegisterType<Inspectors.EnumEditorView>();
            builder.RegisterGeneric(typeof(Inspectors.TextBoxEditorView<>));
        }
    }
}
