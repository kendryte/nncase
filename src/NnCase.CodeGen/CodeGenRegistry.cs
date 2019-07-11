using System;
using System.Collections.Generic;
using System.Text;
using NnCase.CodeGen;
using NnCase.CodeGen.Operators;
using NnCase.IR;
using NnCase.Runtime;

namespace NnCase.CodeGen
{
    public class CodeGenRegistry
    {
        public static CodeGenRegistry Default { get; } = CreateDefaultRegistry();

        private readonly Dictionary<RuntimeTypeHandle, Func<Node, Generator, INodeBody>> _emitters;
        private readonly HashSet<RuntimeTypeHandle> _noRuntimes;

        public CodeGenRegistry()
        {
            _emitters = new Dictionary<RuntimeTypeHandle, Func<Node, Generator, INodeBody>>();
            _noRuntimes = new HashSet<RuntimeTypeHandle>();
        }

        public void Add<T>(Func<T, Generator, INodeBody> emitter)
            where T : Node
        {
            _emitters[typeof(T).TypeHandle] = (n, e) => emitter((T)n, e);
        }

        public void DisableRuntime<T>()
            where T : Node
        {
            _noRuntimes.Add(typeof(T).TypeHandle);
        }

        public bool HasRuntime(Type type)
        {
            return !_noRuntimes.Contains(type.TypeHandle);
        }

        public bool TryInvoke(Node node, Generator generator, out INodeBody body)
        {
            if (_emitters.TryGetValue(node.GetType().TypeHandle, out var e))
            {
                body = e(node, generator);
                return true;
            }

            body = null;
            return _noRuntimes.Contains(node.GetType().TypeHandle);
        }

        private static CodeGenRegistry CreateDefaultRegistry()
        {
            var registry = new CodeGenRegistry();
            DefaultEmitters.Register(registry);
            return registry;
        }
    }
}
