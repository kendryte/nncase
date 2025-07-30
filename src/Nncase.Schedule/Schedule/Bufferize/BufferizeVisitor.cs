// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Reactive;
using System.Text;
using Nncase.Diagnostics;
using Nncase.IR;
using Nncase.TIR;
using Nncase.Utilities;

namespace Nncase.Schedule.Bufferize;

public sealed class BufferizeVisitor : ExprRewriter
{
    private readonly IGrouping<string, PrimFunction> _functions;
    private long _currentRdataStart;
    private long _currentThreadLocalRdataStart;
    private long _currentBlockLocalRdataStart;
    private int _dataBufferId;

    public BufferizeVisitor(IGrouping<string, PrimFunction> functions)
        : base(visitOtherFunctions: true)
    {
        _functions = functions;
    }

    public void Bufferize()
    {
        foreach (var func in _functions)
        {
            Rewrite(func);
        }
    }

    protected override BaseExpr RewriteLeafPrimFunction(PrimFunction func)
    {
        if (!func.SchedResult.IsScheduled)
        {
            (var buffers, var lifetimes) = new LifetimeCollector().Collect(func);
            var scheduleResult = BufferScheduler.Schedule(lifetimes, x => x switch
            {
                MemoryLocation.Rdata => new BufferScheduleOptions(_currentRdataStart),
                MemoryLocation.ThreadLocalRdata => new BufferScheduleOptions(_currentThreadLocalRdataStart),
                MemoryLocation.BlockLocalRdata => new BufferScheduleOptions(_currentBlockLocalRdataStart),
                _ => new BufferScheduleOptions(),
            });

            if (DumpScope.Current.IsEnabled(DumpFlags.Schedule))
            {
                using var dumpScope = new DumpScope(func.Name);
                DumpSchedule(buffers, scheduleResult);
            }

            AssignOutputResult(func, scheduleResult);
            AssignDataResult(func, scheduleResult);
            AssignRdataResult(func, scheduleResult);
            AssignThreadLocalRdataResult(func, scheduleResult);
            AssignBlockLocalRdataResult(func, scheduleResult);

            var bufferReplaces = scheduleResult.SelectMany(x => x.Value.Buffers).ToDictionary(ReferenceEqualityComparer.Instance);
            new BufferReplacer(bufferReplaces).Rewrite(func.Body);
            func.SchedResult.IsScheduled = true;
        }

        return func;
    }

    protected override BaseExpr RewriteLeafCall(Call expr)
    {
        if (expr.Target is PrimFunction func && !func.Name.StartsWith("device_func"))
        {
            if (!func.SchedResult.IsScheduled)
            {
                throw new InvalidOperationException($"Function {func.Name} is not scheduled, please run BufferizePass first.");
            }

            T.CreateBuffer(new TensorType(DataTypes.UInt8, [(long)func.SchedResult.DataUsage]), MemoryLocation.Data, out var dataBuffer, $"data_{_dataBufferId++}");
            var dataVar = new Var("data", TensorType.Scalar(new PointerType(DataTypes.UInt8)));
            var funcParams = func.Parameters.ToArray().Append(dataVar).ToArray();
            var funcArgs = expr.Arguments.ToArray().Append(dataBuffer).ToArray();
            var newFunc = func.With(parameters: funcParams);
            return expr.With(target: newFunc, arguments: funcArgs);
        }

        return expr;
    }

    private void AssignOutputResult(PrimFunction func, IReadOnlyDictionary<MemoryLocation, BufferScheduleResult> scheduleResult)
    {
        if (scheduleResult.TryGetValue(MemoryLocation.Output, out var dataResult))
        {
            func.SchedResult.OutputAlign = Math.Max(8, (ulong)dataResult.Alignment);
            func.SchedResult.OutputUsage = MathUtility.AlignUp((ulong)dataResult.MemoryPoolEnd, func.SchedResult.OutputAlign);
        }
    }

    private void AssignDataResult(PrimFunction func, IReadOnlyDictionary<MemoryLocation, BufferScheduleResult> scheduleResult)
    {
        if (scheduleResult.TryGetValue(MemoryLocation.Data, out var dataResult))
        {
            func.SchedResult.DataAlign = Math.Max(8, (ulong)dataResult.Alignment);
            func.SchedResult.DataUsage = MathUtility.AlignUp((ulong)dataResult.MemoryPoolEnd, func.SchedResult.DataAlign);
        }
    }

    private void AssignRdataResult(PrimFunction func, IReadOnlyDictionary<MemoryLocation, BufferScheduleResult> scheduleResult)
    {
        if (scheduleResult.TryGetValue(MemoryLocation.Rdata, out var rdataResult))
        {
            foreach ((var buffer, var lifetime) in rdataResult.Buffers)
            {
                var constValue = (Const)((Call)buffer.Start)[IR.Buffers.AddressOf.Input];
                var range = new ValueRange<ulong>((ulong)lifetime.Memory.Start, (ulong)lifetime.Memory.Stop);
                func.SchedResult.Rdatas.Add(constValue, range);
            }

            _currentRdataStart = rdataResult.MemoryPoolEnd;
        }
    }

    private void AssignThreadLocalRdataResult(PrimFunction func, IReadOnlyDictionary<MemoryLocation, BufferScheduleResult> scheduleResult)
    {
        if (scheduleResult.TryGetValue(MemoryLocation.ThreadLocalRdata, out var threadLocalRdataResult))
        {
            foreach ((var buffer, var lifetime) in threadLocalRdataResult.Buffers)
            {
                var constValue = (Const)((Call)buffer.Start)[IR.Buffers.AddressOf.Input];
                var range = new ValueRange<ulong>((ulong)lifetime.Memory.Start, (ulong)lifetime.Memory.Stop);
                func.SchedResult.ThreadLocalRdatas.Add(constValue, range);
            }

            _currentThreadLocalRdataStart = threadLocalRdataResult.MemoryPoolEnd;
        }
    }

    private void AssignBlockLocalRdataResult(PrimFunction func, IReadOnlyDictionary<MemoryLocation, BufferScheduleResult> scheduleResult)
    {
        if (scheduleResult.TryGetValue(MemoryLocation.BlockLocalRdata, out var blockLocalRdataResult))
        {
            foreach ((var buffer, var lifetime) in blockLocalRdataResult.Buffers)
            {
                var constValue = (Const)((Call)buffer.Start)[IR.Buffers.AddressOf.Input];
                var range = new ValueRange<ulong>((ulong)lifetime.Memory.Start, (ulong)lifetime.Memory.Stop);
                func.SchedResult.BlockLocalRdatas.Add(constValue, range);
            }

            _currentBlockLocalRdataStart = blockLocalRdataResult.MemoryPoolEnd;
        }
    }

    private void DumpSchedule(TIR.Buffer[] buffers, IReadOnlyDictionary<MemoryLocation, BufferScheduleResult> scheduleResult)
    {
        foreach (var group in buffers.GroupBy(x => x.MemSpan.Buffer.Location))
        {
            var schedule = scheduleResult[group.Key];
            using var wr = new StreamWriter(DumpScope.Current.OpenFile($"{group.Key}.py"), Encoding.UTF8);
            wr.Write(@"from bokeh.models import ColumnDataSource, HoverTool, SingleIntervalTicker, SaveTool, WheelZoomTool, WheelPanTool, ResetTool
from bokeh.palettes import Category20_20 as palette
from bokeh.plotting import figure, show, save
import itertools
from dataclasses import dataclass
from enum import Enum
from typing import List

@dataclass
class Interval():
  start: int
  end: int
  def __str__(self) -> str:
    return f'(start: {self.start}, end {self.end}, size {self.end - self.start})'

class ConstraintsMode(Enum):
  No = 0
  Channel = 1

@dataclass
class ScheduledBuffer():
  name: str
  number: int
  time_interval: Interval
  mem_interval: Interval
  constraints: ConstraintsMode
  shape: List[str]
  stride: List[int]
  inplace: bool

colors = itertools.cycle(palette)

buffers = [
");
            int bufferId = 0;
            foreach (var buffer in group)
            {
                var lifetime = schedule.Buffers[buffer.MemSpan.Buffer];
                var dims = new RankedShape(buffer.Dimensions).Select(x => $"'{x}'");
                var strides = new RankedShape(buffer.Strides).Select(x => $"'{x}'");
                wr.WriteLine($"ScheduledBuffer('{buffer.Name}', {bufferId}, {lifetime.Time}, {lifetime.Memory}, ConstraintsMode.No, [{string.Join(",", dims)}], [{string.Join(",", strides)}], {false}),");
                bufferId++;
            }

            wr.WriteLine(@"]

source = {
    'name': [],
    'x': [],
    'y': [],
    'width': [],
    'height': [],
    'alpha': [],
    'color': [],
    'mem_interval': [],
    'time_interval': [],
    'shape': [],
    'stride': [],
}

y_range_max = 0
x_range_max = 0
color_dict = {}
for buffer in buffers:
  source['name'].append(buffer.name)
  width = buffer.time_interval.end - buffer.time_interval.start
  x = buffer.time_interval.start + (width / 2)
  height = buffer.mem_interval.end - buffer.mem_interval.start
  y = buffer.mem_interval.start + (height / 2)
  y_range_max = max(y_range_max, y)
  x_range_max = max(x_range_max, buffer.time_interval.end)
  source['x'].append(x)
  source['y'].append(y)
  source['width'].append(width)
  source['height'].append(height)
  color = color_dict.get(buffer.name)
  if color == None:
    color = next(colors)
    color_dict[buffer.name] = color
  source['color'].append(color)
  source['alpha'].append(0.2 if buffer.inplace else 1.0)
  source['time_interval'].append(str(buffer.time_interval))
  source['mem_interval'].append(str(buffer.mem_interval))
  source['shape'].append(','.join([str(s) for s in buffer.shape]))
  source['stride'].append(','.join([str(s) for s in buffer.stride]))

source = ColumnDataSource(source)
hover = HoverTool(tooltips=[('name', '@name'), ('time_interval', '@time_interval'), ('mem_interval', '@mem_interval'),
                            ('shape', '@shape'), ('stride', '@stride')])

p = figure(tools=[hover, WheelPanTool(), SaveTool(), WheelZoomTool(), ResetTool()], width=1280, height=720,
           y_range=(0, y_range_max * 1.2), x_range=(-1, x_range_max + 1),
           title='Local Buffer LifeTime (by Steps)')
p.rect(x='x', y='y', width='width', height='height', fill_color='color', legend_field='name', fill_alpha='alpha', source=source)
p.xaxis.axis_label = 'Time (steps)'
p.outline_line_color = None");

            wr.WriteLine($@"
save(p, filename='{group.Key}.html')
show(p)");
        }
    }

    private sealed class BufferReplacer : ExprRewriter
    {
        private readonly IReadOnlyDictionary<TIR.PhysicalBuffer, BufferLifetime> _buffers;

        public BufferReplacer(IReadOnlyDictionary<TIR.PhysicalBuffer, BufferLifetime> buffers)
        {
            _buffers = buffers;
        }

        protected override BaseExpr RewriteLeafPhysicalBuffer(TIR.PhysicalBuffer expr)
        {
            if (_buffers.TryGetValue(expr, out var lifetime))
            {
                var start = Tensor.FromScalar((ulong)lifetime.Memory.Start);
                return expr.With(start: start);
            }

            return expr;
        }
    }
}
