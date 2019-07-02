using System;
using System.Collections.Generic;
using System.Linq;
using DotBuilder.Attributes;

namespace DotBuilder.Statements
{
    public class GraphBase
    {
        private readonly AttributesFor<IGraphAttribute> _attributes = new AttributesFor<IGraphAttribute>("graph");
        private readonly string _graphType;
        private readonly string _name;
        private readonly List<IStatement> _statements = new List<IStatement>();

        protected GraphBase(string graphType, string name)
        {
            _graphType = graphType;
            _name = name;
        }

        public GraphBase WithGraphAttributesOf(params IGraphAttribute[] attributes)
        {
            _attributes.WithAttributesOf(attributes);
            return this;
        }

        public GraphBase WithGraphAttributesOf(IEnumerable<IGraphAttribute> attributes)
        {
            _attributes.WithAttributesOf(attributes);
            return this;
        }
        public GraphBase WithNodeAttributesOf(params INodeAttribute[] attributes)
        {
            Containing(new AttributesFor<INodeAttribute>("node").WithAttributesOf(attributes));
            return this;
        }

        public GraphBase WithEdgeAttributesOf(params IEdgeAttribute[] attributes)
        {
            Containing(new AttributesFor<IEdgeAttribute>("edge").WithAttributesOf(attributes));
            return this;
        }

        public GraphBase WithNodeAttributesOf(IEnumerable<INodeAttribute> attributes)
        {
            Containing(new AttributesFor<INodeAttribute>("node").WithAttributesOf(attributes));
            return this;
        }
        public GraphBase WithEdgeAttributesOf(IEnumerable<IEdgeAttribute> attributes)
        {
            Containing(new AttributesFor<IEdgeAttribute>("edge").WithAttributesOf(attributes));
            return this;
        }

        public GraphBase Containing(params IStatement[] statements)
        {
            _statements.AddRange(statements);
            return this;
        }

        public GraphBase Containing(IEnumerable<IStatement> statements)
        {
            _statements.AddRange(statements);
            return this;
        }

        [Obsolete]
        public GraphBase Of(params IGraphAttribute[] attributes) => WithGraphAttributesOf(attributes);

        [Obsolete]
        public GraphBase Of(IEnumerable<IGraphAttribute> attributes) => WithGraphAttributesOf(attributes);

        [Obsolete]
        public GraphBase With(params IStatement[] statements) => Containing(statements);

        [Obsolete]
        public GraphBase With(IEnumerable<IStatement> statements) => Containing(statements);


        public string Render()
        {
            return $"{_graphType} \"{_name}\" {{ \n" +
                   $"{_attributes.Render()}\n" +
                   $"{_statements.Aggregate("", (x, s) => x + s.Render() + "\n")} }}";
        }
    }
}