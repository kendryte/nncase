// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Nncase.Diagnostics;
using Nncase.IR;
using Nncase.Schedule;
using Nncase.Schedule.Bufferize;
using Nncase.TIR;
using Nncase.Utilities;

namespace Nncase.Passes.Transforms;

public sealed class BufferizePass : FunctionPass
{
    protected override Task<BaseFunction> RunCoreAsync(BaseFunction input, RunPassContext context)
    {
        if (input is PrimFunction primFunc)
        {
            Bufferize(primFunc);
        }

        return Task.FromResult(input);
    }

    private void Bufferize(PrimFunction func)
    {
        var lifetimes = new LifetimeCollector().Collect(func);
        var scheduleResult = BufferScheduler.Schedule(lifetimes);
        if (DumpScope.Current.IsEnabled(DumpFlags.Schedule))
        {
            DumpSchedule(scheduleResult);
        }

        AssignDataResult(func, scheduleResult);
        AssignRdataResult(func, scheduleResult);
        AssignLocalRdataResult(func, scheduleResult);

        var bufferReplaces = scheduleResult.SelectMany(x => x.Value.Buffers).ToDictionary(
            x => x.Buffer,
            (IEqualityComparer<TIR.Buffer>)ReferenceEqualityComparer.Instance);
        new BufferReplacer(bufferReplaces).Rewrite(func.Body);
        func.SchedResult.IsScheduled = true;
    }

    private void AssignDataResult(PrimFunction func, IReadOnlyDictionary<MemoryLocation, BufferScheduleResult> scheduleResult)
    {
        if (scheduleResult.TryGetValue(MemoryLocation.Data, out var dataResult))
        {
            func.SchedResult.DataAlign = (ulong)dataResult.Alignment;
            func.SchedResult.DataUsage = (ulong)dataResult.MemoryPoolSize;
        }
    }

    private void AssignRdataResult(PrimFunction func, IReadOnlyDictionary<MemoryLocation, BufferScheduleResult> scheduleResult)
    {
        if (scheduleResult.TryGetValue(MemoryLocation.Rdata, out var rdataResult))
        {
            foreach (var lifetime in rdataResult.Buffers)
            {
                var constValue = (Const)((Call)lifetime.Buffer.MemSpan.Start)[IR.Buffers.DDrOf.Input];
                var range = new ValueRange<ulong>((ulong)lifetime.Memory.Start, (ulong)lifetime.Memory.Stop);
                func.SchedResult.Rdatas.Add(constValue, range);
            }
        }
    }

    private void AssignLocalRdataResult(PrimFunction func, IReadOnlyDictionary<MemoryLocation, BufferScheduleResult> scheduleResult)
    {
        if (scheduleResult.TryGetValue(MemoryLocation.ThreadLocalRdata, out var localRdataResult))
        {
            foreach (var lifetime in localRdataResult.Buffers)
            {
                var constValue = (Const)((Call)lifetime.Buffer.MemSpan.Start)[IR.Buffers.DDrOf.Input];
                var range = new ValueRange<ulong>((ulong)lifetime.Memory.Start, (ulong)lifetime.Memory.Stop);
                func.SchedResult.LocalRdatas.Add(constValue, range);
            }
        }
    }

    private void DumpSchedule(IReadOnlyDictionary<MemoryLocation, BufferScheduleResult> scheduleResult)
    {
        foreach (var locationResult in scheduleResult)
        {
            using var wr = new StreamWriter(DumpScope.Current.OpenFile($"{locationResult.Key}.py"), Encoding.UTF8);
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
            foreach (var lifetime in locationResult.Value.Buffers)
            {
                var dims = new Shape(lifetime.Buffer.Dimensions).Select(x => $"'{x}'");
                var strides = new Shape(lifetime.Buffer.Strides).ToValueArray();
                wr.WriteLine($"ScheduledBuffer('{lifetime.Buffer.Name}', {bufferId}, {lifetime.Time}, {lifetime.Memory}, ConstraintsMode.No, [{string.Join(",", dims)}], [{string.Join(",", strides)}], {false}),");
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
save(p, filename='{locationResult.Key.ToString()}.html')
show(p)");
        }
    }

    private sealed class BufferReplacer : ExprRewriter
    {
        private readonly IReadOnlyDictionary<TIR.Buffer, BufferLifetime> _buffers;

        public BufferReplacer(IReadOnlyDictionary<TIR.Buffer, BufferLifetime> buffers)
        {
            _buffers = buffers;
        }

        protected override Expr RewriteLeafBuffer(TIR.Buffer expr)
        {
            if (_buffers.TryGetValue(expr, out var lifetime))
            {
                var start = Tensor.FromPointer((ulong)lifetime.Memory.Start, expr.ElemType);
                if (start.ElementType is PointerType { ElemType: ReferenceType { ElemType: IR.NN.PagedAttentionKVCacheType ntype } } && expr.ElemType is ReferenceType { ElemType: IR.NN.PagedAttentionKVCacheType oldType })
                {
                    ntype.Config = oldType.Config;
                }

                var memSpan = expr.MemSpan.With(start: start);
                return expr.With(memSpan: memSpan);
            }

            return expr;
        }
    }
}
