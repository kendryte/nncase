using System;
using System.Collections.Generic;
using System.Linq;
using DotBuilder.Attributes;

namespace DotBuilder.Statements
{
    public interface IStatement
    {
        string Render();
    }

    public abstract class Statement<TS, TA> : IStatement where TA : IAttribute
    {
        private readonly List<TA> _attributes = new List<TA>();

        public virtual string Render()
        {
            var attributes = $"{string.Join(",", _attributes.GroupBy(x => x.GetType()).SelectMany(Combine).Select(x => x.Render()))}";
            return attributes.Length > 0 ? $"[{attributes}]" : "";
        }

        private IAttribute[] Combine(IGrouping<Type, TA> arg)
        {
            if (typeof(IMustBeCombined).IsAssignableFrom(arg.Key))
            {
                var combinedValue = string.Join(",", arg.Select(x => x.Value));
                return new []{ (IAttribute) new Attrib(arg.First().Name,combinedValue)};
            }

            return arg.Cast<IAttribute>().ToArray();
        }

        public Statement<TS, TA> WithAttributesOf(params TA[] attributes)
        {
            _attributes.AddRange(attributes);
            return this;
        }

        public Statement<TS, TA> WithAttributesOf(IEnumerable<TA> attributes)
        {
            _attributes.AddRange(attributes);
            return this;
        }

        [Obsolete]
        public Statement<TS, TA> Of(params TA[] attributes) => WithAttributesOf(attributes);
        [Obsolete]
        public Statement<TS, TA> Of(IEnumerable<TA> attributes) => WithAttributesOf(attributes);
    }
}