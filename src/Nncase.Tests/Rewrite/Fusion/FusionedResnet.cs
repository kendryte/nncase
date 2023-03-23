// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.IO;
using Nncase.IR;
using Nncase.Passes;
using Xunit;
using static Nncase.IR.F.Tensors;
using static Nncase.PatternMatch.Utility;

namespace Nncase.Tests.ReWrite.FusionTest;

internal interface IForwardable
{
    Expr Forward(params Expr[] inputs);
}

internal sealed record ForwardFusion(Func<Expr[], Fusion> Creator) : IForwardable
{
    public Expr Forward(params Expr[] inputs)
    {
        var fusion = Creator(inputs);
        return new Call(fusion, inputs);
    }

    public static ForwardFusion Binary(BinaryOp binaryOp)
    {
        var creator = (Expr[] inputs) =>
        {
            if (inputs.Length != 2)
            {
                throw new NotSupportedException();
            }

            var a = inputs[0];
            var b = inputs[1];
            if (a.CheckedType is null)
            {
                CompilerServices.InferenceType(a);
            }

            if (b.CheckedType is null)
            {
                CompilerServices.InferenceType(b);
            }

            var in_a = new Var(a.CheckedType!);
            var in_b = new Var(b.CheckedType!);
            return new Fusion("BinaryFusion", Callable.StackVMModuleKind, IR.F.Math.Binary(BinaryOp.Add, in_a, in_b), new[] { in_a, in_b });
        };
        return new(creator);
    }

    public static ForwardFusion Conv3x3(int in_planes, int out_planes, int stride = 1, int groups = 1, int dilation = 1)
    {
        var creator = (Expr[] inputs) =>
        {
            if (inputs.Length != 1)
            {
                throw new NotSupportedException();
            }

            var input = inputs[0];
            if (input.CheckedType is null)
            {
                CompilerServices.InferenceType(input);
            }

            var v_input = new Var(input.CheckedType!);
            var weights = IR.F.Random.Normal(DataTypes.Float32, 0, 1, 1, new[] { out_planes, v_input.CheckedShape[1].FixedValue, 3, 3 }).Evaluate().AsTensor();
            var bias = IR.F.Random.Normal(DataTypes.Float32, 0, 1, 1, new[] { out_planes }).Evaluate().AsTensor();

            return new Fusion("Conv3x3Fusion", Callable.StackVMModuleKind, IR.F.NN.Conv2D(v_input, weights, bias, new[] { stride, stride }, new[,] { { dilation, dilation }, { dilation, dilation } }, new[] { dilation, dilation }, PadMode.Constant, groups), new[] { v_input });
        };
        return new(creator);
    }

    public static ForwardFusion Conv1x1(int in_planes, int out_planes, int stride = 1)
    {
        var creator = (Expr[] inputs) =>
          {
              if (inputs.Length != 1)
              {
                  throw new NotSupportedException();
              }

              var input = inputs[0];
              if (input.CheckedType is null)
              {
                  CompilerServices.InferenceType(input);
              }

              var v_input = new Var(input.CheckedType!);
              var weights = IR.F.Random.Normal(DataTypes.Float32, 0, 1, 1, new[] { out_planes, v_input.CheckedShape[1].FixedValue, 1, 1 }).Evaluate().AsTensor();
              var bias = IR.F.Random.Normal(DataTypes.Float32, 0, 1, 1, new[] { out_planes }).Evaluate().AsTensor();
              return new Fusion("Conv1x1Fusion", Callable.StackVMModuleKind, IR.F.NN.Conv2D(v_input, weights, bias, new[] { stride, stride }, new[,] { { 0, 0 }, { 0, 0 }, }, new[] { 1, 1 }, PadMode.Constant, 1), new[] { v_input });
          };
        return new(creator);
    }
}

internal sealed record ForwardSequential(IForwardable[] Array) : IForwardable
{
    public Expr Forward(params Expr[] inputs)
    {
        if (inputs.Length != 1)
        {
            throw new NotSupportedException();
        }

        var x = inputs[0];
        Expr last = x;
        foreach (var f in Array)
        {
            last = f.Forward(last);
        }

        return last;
    }
}

[AttributeUsage(AttributeTargets.Class, AllowMultiple = false)]
public sealed class ExpansionAttribute : Attribute
{
    private readonly int _expansion;

    public ExpansionAttribute(int expansion)
    {
        _expansion = expansion;
    }

    public int Expansion => _expansion;
}

[Expansion(1)]
internal sealed class BasicBlock : IForwardable
{
    private readonly ForwardFusion _conv1;
    private readonly ForwardFusion _conv2;
    private readonly ForwardFusion? _downsample;
    private readonly ForwardFusion _binaryAdd;
    private readonly int _stride;

    public BasicBlock(int inplanes, int planes, int stride = 1, ForwardFusion? downsample = null, int groups = 1, int base_width = 64, int dilation = 1)
    {
        if (groups != 1 || base_width != 64)
        {
            throw new System.NotSupportedException();
        }

        // raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if (dilation > 1)
        {
            // raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
            throw new System.NotSupportedException();
        }

        // # Both this.conv1 and this.downsample layers downsample the input when stride != 1
        _conv1 = ForwardFusion.Conv3x3(inplanes, planes, stride);
        _conv2 = ForwardFusion.Conv3x3(planes, planes);
        _downsample = downsample;
        _stride = stride;
        _binaryAdd = ForwardFusion.Binary(BinaryOp.Add);
    }

    public Expr Forward(params Expr[] inputs)
    {
        if (inputs.Length != 1)
        {
            throw new NotSupportedException();
        }

        var x = inputs[0];
        var identity = x;
        var y = _conv1.Forward(x);
        y = _conv2.Forward(y);
        if (_downsample is not null)
        {
            identity = _downsample.Forward(x);
        }

        y = _binaryAdd.Forward(y, identity);
        return y;
    }
}

[Expansion(4)]
internal sealed class Bottleneck : IForwardable
{
    // # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(this.conv2)
    // # while original implementation places the stride at the first 1x1 convolution(this.conv1)
    // # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    // # This variant is also known as ResNet V1.5 and improves accuracy according to
    // # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.
    private readonly ForwardFusion _conv1;
    private readonly ForwardFusion _conv2;
    private readonly ForwardFusion _conv3;
    private readonly ForwardFusion? _downsample;
    private readonly ForwardFusion _binaryAdd;
    private readonly int _stride;

    public Bottleneck(
        int inplanes,
        int planes,
        int stride = 1,
        ForwardFusion? downsample = null,
        int groups = 1,
        int base_width = 64,
        int dilation = 1)
    {
        var width = (int)(planes * (base_width / 64.0)) * groups;

        // Both this.conv2 and this.downsample layers downsample the input when stride != 1
        _conv1 = ForwardFusion.Conv1x1(inplanes, width);
        _conv2 = ForwardFusion.Conv3x3(width, width, stride, groups, dilation);
        var expansion = (ExpansionAttribute)Attribute.GetCustomAttribute(typeof(Bottleneck), typeof(ExpansionAttribute))!;
        _conv3 = ForwardFusion.Conv1x1(width, planes * expansion.Expansion);
        _downsample = downsample;
        _stride = stride;
        _binaryAdd = ForwardFusion.Binary(BinaryOp.Add);
    }

    public Expr Forward(params Expr[] inputs)
    {
        if (inputs.Length != 1)
        {
            throw new NotSupportedException();
        }

        var x = inputs[0];
        var identity = x;
        var y = _conv1.Forward(x);
        y = _conv2.Forward(y);
        y = _conv3.Forward(y);

        if (_downsample is not null)
        {
            identity = _downsample.Forward(x);
        }

        y = _binaryAdd.Forward(y, identity);
        return y;
    }
}

internal sealed class ResNet
{
    private readonly int _groups;
    private readonly int _baseWidth;
    private readonly ForwardFusion _conv1;
    private readonly ForwardFusion _maxpool;
    private readonly IForwardable _layer1;
    private readonly IForwardable _layer2;
    private readonly IForwardable _layer3;
    private readonly IForwardable _layer4; // avgpool, fc;
    private int _inplanes;
    private int _dilation;

    public ResNet(
        Type block,
        int[] layers,
        int num_classes = 1000,
        bool zero_init_residual = false,
        int groups = 1,
        int width_per_group = 64,
        bool[]? replace_stride_with_dilation = null)
    {
        _inplanes = 64;
        _dilation = 1;
        if (replace_stride_with_dilation is null)
        {
            // # each element in the tuple indicates if we should replace
            // # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = new[] { false, false, false };
        }

        _groups = groups;
        _baseWidth = width_per_group;
        var conv1_creator = (Expr[] inputs) =>
        {
            if (inputs.Length != 1)
            {
                throw new NotSupportedException();
            }

            var input = inputs[0];
            if (input.CheckedType is null)
            {
                CompilerServices.InferenceType(input);
            }

            var v_input = new Var(input.CheckedType!);
            var weights = IR.F.Random.Normal(DataTypes.Float32, 0, 1, 1, new[] { 64, v_input.CheckedShape[1].FixedValue, 7, 7 }).Evaluate().AsTensor();
            var bias = IR.F.Random.Normal(DataTypes.Float32, 0, 1, 1, new[] { 64 }).Evaluate().AsTensor();
            return new Fusion(
                "Conv3x3Fusion",
                Callable.StackVMModuleKind,
                IR.F.NN.Conv2D(
                    v_input,
                    weights,
                    bias,
                    new[] { 2, 2 },
                    new[,]
                    {
                        { 3, 3 },
                        { 3, 3 },
                    },
                    new[] { 1, 1 },
                    PadMode.Constant,
                    groups),
                new[] { v_input });
        };
        _conv1 = new(conv1_creator);

        var maxpool_creator = (Expr[] inputs) =>
        {
            if (inputs.Length != 1)
            {
                throw new NotSupportedException();
            }

            var input = inputs[0];
            if (input.CheckedType is null)
            {
                CompilerServices.InferenceType(input);
            }

            var v_input = new Var(input.CheckedType!);

            return new Fusion("ReduceWindowFusion", Callable.StackVMModuleKind, IR.F.NN.ReduceWindow2D(ReduceOp.Max, v_input, 0.0f, new[] { 3, 3 }, new[] { 2, 2 }, new[,] { { 1, 1 }, { 1, 1 } }, new[] { 1, 1 }, false, false), new[] { v_input });
        };
        _maxpool = new(maxpool_creator);
        _layer1 = Make_layer(block, 64, layers[0]);
        _layer2 = Make_layer(block, 128, layers[1], stride: 2, dilate: replace_stride_with_dilation[0]);
        _layer3 = Make_layer(block, 256, layers[2], stride: 2, dilate: replace_stride_with_dilation[1]);
        _layer4 = Make_layer(block, 512, layers[3], stride: 2, dilate: replace_stride_with_dilation[2]);

        // this.avgpool = nn.AdaptiveAvgPool2d((1, 1));
        // this.fc = nn.Linear(512 * block.expansion, num_classes);
    }

    public Expr Forward(params Expr[] inputs)
    {
        if (inputs.Length != 1)
        {
            throw new NotSupportedException();
        }

        var x = inputs[0];
        x = _conv1.Forward(x);
        x = _maxpool.Forward(x);
        x = _layer1.Forward(x);
        x = _layer2.Forward(x);
        x = _layer3.Forward(x);
        x = _layer4.Forward(x);
        return x;
    }

    private IForwardable Make_layer(Type block, int planes, int blocks, int stride = 1, bool dilate = false)
    {
        IForwardable? downsample = null;
        var previous_dilation = _dilation;

        if (dilate)
        {
            _dilation *= stride;
            stride = 1;
        }

        // block.GetField("expansion").GetValue()
        var expansion = ((ExpansionAttribute)Attribute.GetCustomAttribute(block, typeof(ExpansionAttribute))!).Expansion;
        if (stride != 1 || _inplanes != planes * expansion)
        {
            downsample = ForwardFusion.Conv1x1(_inplanes, planes * expansion, stride);
        }

        var layers = new List<IForwardable>();

        layers.Add(
          (IForwardable)Activator.CreateInstance(
              block,
              new object?[] { _inplanes, planes, stride, downsample, _groups, _baseWidth, previous_dilation })!);
        _inplanes = planes * expansion;
        for (int i = 1; i < blocks; i++)
        {
            layers.Add(
              (IForwardable)Activator.CreateInstance(
                  block,
                  new object?[] { _inplanes, planes, 1, null, _groups, _baseWidth, _dilation })!);
        }

        return new ForwardSequential(layers.ToArray());
    }
}
