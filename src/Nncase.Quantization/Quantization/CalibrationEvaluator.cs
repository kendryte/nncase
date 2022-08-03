// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Nncase.Evaluator;
using Nncase.IR;
using Nncase.Transform;

namespace Nncase.Quantization;

public class CalibrationEvaluator
{
    private readonly IReadOnlyDictionary<Var, IValue> _inputs;
    private readonly IEnumerable<ENode> _awareEnodes;
    private readonly Dictionary<ENode, IValue> _values = new(ReferenceEqualityComparer.Instance);
    private readonly Dictionary<EClass, IValue> _eclassValues = new();
    private readonly RunPassOptions _passOptions;
    private StreamWriter _dumpWriter;

    public CalibrationEvaluator(IReadOnlyDictionary<Var, IValue> inputs, IEnumerable<ENode> awareEnodes, RunPassOptions passOptions)
    {
        _inputs = inputs;
        _awareEnodes = awareEnodes;
        _passOptions = passOptions;
        _dumpWriter = StreamWriter.Null;
    }

    public IReadOnlyDictionary<ENode, Tensor> Evaluate()
    {
        if (_passOptions.DumpLevel >= 4)
        {
            if (!Directory.Exists(_passOptions.PassDumpDir))
                Directory.CreateDirectory(_passOptions.PassDumpDir);
            var fileStream = File.Open(Path.Combine(_passOptions.PassDumpDir, "calibration_evaluator.il"), FileMode.Create, FileAccess.Write);
            _dumpWriter = new StreamWriter(fileStream);
            _dumpWriter.AutoFlush = true;
        }

        bool completed;
        var awareTensors = new Dictionary<ENode, Tensor>();

        do
        {
            completed = true;
            var oldValues = _values.Count;
            foreach (var enode in _awareEnodes)
            {
                var value = Visit(enode);
                if (value == null)
                {
                    completed = false;
                }
                else
                {
                    awareTensors[enode] = value.AsTensor();
                }
            }

            if (_awareEnodes.Count() > 0 && _values.Count == oldValues)
            {
                throw new InvalidOperationException("Endless evaluation found.");
            }
        }
        while (!completed);
        _dumpWriter.Close();
        return awareTensors;
    }

    private IValue? Visit(EClass eclass)
    {
        if (!_eclassValues.TryGetValue(eclass, out var value))
        {
            foreach (var enode in eclass.Nodes)
            {
                var enodeValue = Visit(enode);
                if (enodeValue != null)
                {
                    value = enodeValue;
                    _eclassValues.Add(eclass, value);
                    break;
                }
            }
        }

        return value;
    }

    private IValue? Visit(ENode enode)
    {
        return enode.Expr switch
        {
            Var var => Visit(enode, var),
            TensorConst con => Visit(enode, con),
            TupleConst con => Visit(enode, con),
            Function func => Visit(enode, func),
            Call call => Visit(enode, call),
            IR.Tuple tuple => Visit(enode, tuple),
            Op op => Visit(enode, op),
            Marker marker => Visit(enode, marker),
            None none => Visit(enode, none),
            _ => throw new ArgumentException("Unsupported expression type."),
        };
    }

    private bool shapeChecker(Shape current, Shape target)
    {
        if (current.Count != target.Count)
            return false;
        return current.Zip(target).All(p => p.Item2.IsUnknown ? true : p.Item2.FixedValue == p.Item1.FixedValue);
    }

    private bool typeChecker(IRType cur_type, IRType target_type) => (cur_type, target_type) switch
    {
        (TensorType a, TensorType b) => a.DType == b.DType && shapeChecker(a.Shape, b.Shape),
        (TupleType a, TupleType b) => a.Zip(b).All(p => typeChecker(p.Item1, p.Item2)),
        (_, _) => true,
    };

    private IValue? Visit(ENode enode, Var var)
    {
        return VisitLeaf(enode, () =>
        {
            var value = _inputs[var];
            if (!new TypePattern(cur => typeChecker(cur, var.CheckedType!), "Var Type Checker").MatchLeaf(value.Type))
                throw new InvalidOperationException("Feed Value Is Invalid!");
            return value;
        });
    }

    private IValue Visit(ENode enode, TensorConst tc)
    {
        return VisitLeaf(enode, () => Value.FromConst(tc));
    }

    private IValue Visit(ENode enode, Op op)
    {
        return NoneValue.Default;
    }

    private IValue Visit(ENode enode, Function func)
    {
        return NoneValue.Default;
    }

    private IValue? Visit(ENode enode, TupleConst tc)
    {
        return VisitLeaf(enode, () => Value.FromConst(tc));
    }

    private IValue? Visit(ENode enode, IR.Tuple tuple)
    {
        return Visit(enode, values => new TupleValue(values));
    }

    private IValue? Visit(ENode enode, Marker marker)
    {
        return Visit(enode, costs => costs[0]);
    }

    private IValue? Visit(ENode enode, None none)
    {
        return NoneValue.Default;
    }

    private IValue? Visit(ENode enode, Call call)
    {
        System.Console.WriteLine($"Visit({enode}, {call.Target.GetType().Name})");
        return Visit(enode, costs =>
        {
            IValue? value = null;
            foreach (var targetEnode in enode.Children[0].Nodes)
            {
                if (targetEnode.Expr is Op op)
                {
                    var context = new EGraphOpEvaluateContext(call, costs.Skip(1).ToArray());
                    if (_passOptions.DumpLevel >= 4)
                        _dumpWriter.Write($"{op.GetType().Name}({string.Join(",", context.Arguments.Select(v => v.ToString()))})");
                    value = CompilerServices.EvaluateOp(op, context);
                    if (_passOptions.DumpLevel >= 4)
                        _dumpWriter.WriteLine($" => {value.ToString()}");
                }
                else
                {
                    Debug.Assert(targetEnode.Expr is Function);
                    value = Visit(targetEnode.Children[0]);
                }

                if (value != null)
                {
                    return value;
                }
            }

            return null;
        });
    }

    private IValue VisitLeaf(ENode enode, Func<IValue> valueGetter)
    {
        if (!_values.TryGetValue(enode, out var value))
        {
            value = valueGetter();
            _values[enode] = value;
        }

        return value;
    }

    private IValue? Visit(ENode enode, Func<IValue[], IValue?> valueGetter)
    {
        if (!_values.TryGetValue(enode, out var value))
        {
            var values = new IValue[enode.Children.Count];
            for (int i = 0; i < values.Length; i++)
            {
                var childValue = Visit(enode.Children[i]);
                if (childValue != null)
                {
                    values[i] = childValue;
                }
                else
                {
                    return null;
                }
            }

            value = valueGetter(values);
            if (value != null)
            {
                _values.Add(enode, value);
            }
        }

        return value;
    }

    private sealed class EGraphOpEvaluateContext : IEvaluateContext
    {
        public readonly IValue[] Arguments;

        public EGraphOpEvaluateContext(Call currentCall, IValue[] arguments)
        {
            CurrentCall = currentCall;
            Arguments = arguments;
        }

        public Call CurrentCall { get; }

        public IValue GetArgumentValue(Op op, ParameterInfo parameter)
        {
            var index = op.GetType() == parameter.OwnerType
                ? parameter.Index
                : throw new ArgumentOutOfRangeException($"Operator {op} doesn't have parameter: {parameter.Name}.");
            return Arguments[index];
        }
    }
}
