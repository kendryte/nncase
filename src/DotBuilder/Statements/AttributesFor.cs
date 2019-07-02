using System;
using DotBuilder.Attributes;

namespace DotBuilder.Statements
{
    public class AttributesFor<T> : Statement<AttributesFor<T>, T> where T : IAttribute
    {
        private readonly string _type;

        internal AttributesFor(string type)
        {
            _type = type;
        }

        public override string Render()
        {
            var render = base.Render();
            if (string.IsNullOrEmpty(render))
                return string.Empty;
            else
                return $"{_type} {render}";
        }
    }

    public static class AttributesFor
    {
        [Obsolete]
        public static AttributesFor<INodeAttribute> Node => new AttributesFor<INodeAttribute>("node");
        [Obsolete]
        public static AttributesFor<IEdgeAttribute> Edge => new AttributesFor<IEdgeAttribute>("edge");
    }
}