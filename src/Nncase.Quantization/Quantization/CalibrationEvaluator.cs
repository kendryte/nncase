// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Nncase.Diagnostics;
using Nncase.Evaluator;
using Nncase.IR;
using Nncase.Passes;

namespace Nncase.Quantization;

public class CalibrationEvaluator : IDisposable
{
    private readonly IReadOnlyDictionary<Var, IValue> _inputs;
    private readonly IEnumerable<ENode> _awareEnodes;
    private readonly Dictionary<ENode, IValue> _values = new(ReferenceEqualityComparer.Instance);
    private readonly Dictionary<EClass, IValue> _eclassValues = new();
    private readonly StreamWriter? _dumpWriter;

    public CalibrationEvaluator(IReadOnlyDictionary<Var, IValue> inputs, IEnumerable<ENode> awareEnodes)
    {
        _inputs = inputs;
        _awareEnodes = awareEnodes;
        _dumpWriter = DumpScope.Current.IsEnabled(DumpFlags.Calibration)
            ? new StreamWriter(DumpScope.Current.OpenFile("calibration_evaluator.il")) { AutoFlush = true }
            : null;
    }

    public IReadOnlyDictionary<ENode, Tensor> Evaluate()
    {
        bool completed;
        var awareTensors = new Dictionary<ENode, Tensor>(ReferenceEqualityComparer.Instance);

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
                    if (value != Value.None)
                    {
                        awareTensors[enode] = value.AsTensor();
                    }
                    else
                    {
                        awareTensors[enode] = Enumerable.Repeat(0, 0).Select(x => (float)x).ToArray();
                        _values.TryAdd(enode, Value.None);
                    }
                }
            }

            if (_awareEnodes.Any() && _values.Count == oldValues)
            {
                throw new InvalidOperationException("Endless evaluation found.");
            }
        }
        while (!completed);
        return awareTensors;
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        _dumpWriter?.Dispose();
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
            If @if => Visit(enode, @if),
            _ => throw new ArgumentException("Unsupported expression type."),
        };
    }

    private bool ShapeChecker(Shape current, Shape target)
    {
        if (current.Count != target.Count)
        {
            return false;
        }

        return current.Zip(target).All(p => p.Second.IsUnknown ? true : p.Second.FixedValue == p.First.FixedValue);
    }

    private bool TypeChecker(IRType cur_type, IRType target_type) => (cur_type, target_type) switch
    {
        (TensorType a, TensorType b) => a.DType == b.DType && ShapeChecker(a.Shape, b.Shape),
        (TupleType a, TupleType b) => a.Zip(b).All(p => TypeChecker(p.First, p.Second)),
        (_, _) => true,
    };

    private IValue? Visit(ENode enode, Var var)
    {
        return VisitLeaf(enode, () =>
        {
            var value = _inputs[var];
            if (!new TypePattern(cur => TypeChecker(cur, var.CheckedType!), "Var Type Checker").MatchLeaf(value.Type))
            {
                throw new InvalidOperationException($"Feed Value Is Invalid, need {var.CheckedType} but get {value.Type}!");
            }

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

    private IValue? Visit(ENode enode, If expr)
    {
        return Visit(enode, values =>
        {
            return values[^3].AsTensor().ToScalar<bool>()
                ? values[^2]
                : values[^1];
        });
    }

    private IValue? Visit(ENode enode, Call call)
    {
        return Visit(enode, costs =>
        {
            IValue? value = null;
            foreach (var targetEnode in enode.Children[0].Nodes)
            {
                if (targetEnode.Expr is Op op)
                {
                    var context = new EGraphOpEvaluateContext(call, costs.Skip(1).ToArray());
                    _dumpWriter?.Write($"{op.GetType().Name}({string.Join(",", context.Arguments.Select(v => v.ToString()))})");

                    if (op.CanFoldConstCall)
                    {
                        value = CompilerServices.EvaluateOp(op, context);
                    }
                    else
                    {
                        value = NoneValue.Default;
                    }

                    _dumpWriter?.WriteLine($" => {value}");
                }
                else
                {
                    Trace.Assert(targetEnode.Expr is Function);
                    value = CompilerServices.Evaluate(((Function)targetEnode.Expr).Body, _inputs);
                    return value;
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

        if (enode.Expr is Nncase.IR.Marker && ((Marker)enode.Expr).MixQuantInfo?.QuantParameter != null && ((Marker)enode.Expr).MixQuantInfo!.QuantParameter.Count != 0 && value != null)
        {
            var valueArray = value.AsTensor().ToArray<float>();
            int index = 0;

            int size = 0;
            if (((Marker)enode.Expr).MixQuantInfo!.QuantParameter.Count != 1)
            {
                size = value.AsTensor().Shape[1].FixedValue * value.AsTensor().Shape[2].FixedValue * value.AsTensor().Shape[3].FixedValue;
            }

            for (int i = 0; i < valueArray.Length; i++)
            {
                if (((Marker)enode.Expr).MixQuantInfo!.QuantParameter.Count != 1)
                {
                    index = i / size;
                }

                var valueArrayQuant = Math.Round((valueArray[i] / (double)((Marker)enode.Expr).MixQuantInfo!.QuantParameter[index].Scale) + ((Marker)enode.Expr).MixQuantInfo!.QuantParameter[index].ZeroPoint);
                var valueArrayDeQuant = ((float)valueArrayQuant - ((Marker)enode.Expr).MixQuantInfo!.QuantParameter[index].ZeroPoint) * (double)((Marker)enode.Expr).MixQuantInfo!.QuantParameter[index].Scale;
                valueArray[i] = (float)valueArrayDeQuant;
            }

            value = Value.FromTensor(Tensor.From<float>(valueArray, value.AsTensor().Shape));
        }

        return value;
    }

    private sealed class EGraphOpEvaluateContext : IEvaluateContext
    {
        public EGraphOpEvaluateContext(Call currentCall, IValue[] arguments)
        {
            CurrentCall = currentCall;
            Arguments = arguments;
        }

        public IValue[] Arguments { get; }

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
