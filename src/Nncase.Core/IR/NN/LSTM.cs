// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.
// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.
// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.
// // Copyright (c) Canaan Inc. All rights reserved.
// // Licensed under the Apache license. See LICENSE file in the project root for full license information.
//
// using System;
// using System.Collections.Generic;
// using System.Collections.Immutable;
// using System.Linq;
// using System.Text;
// using System.Threading.Tasks;
// using Nncase.PatternMatch;
// using static Nncase.IR.TypePatternUtility;
//
// namespace Nncase.IR.NN
// {
//     public enum lstm_direction
//     {
//         kForward,
//         kReverse,
//         kBidirectional
//     }
//
//     [PatternFunctionalGenerator]
//     public sealed record LSTM(lstm_direction direction, String framework) : Op
//     {
//         /// <summary>
//         /// Gets input.
//         /// </summary>
//         public static readonly ParameterInfo Input = new(typeof(LSTM), 0, "input", HasDataType(DataTypes.Float32));
//
//         /// <summary>
//         /// Gets w.
//         /// </summary>
//         public static readonly ParameterInfo W = new(typeof(LSTM), 1, "w");
//
//         /// <summary>
//         /// Gets r.
//         /// </summary>
//         public static readonly ParameterInfo R = new(typeof(LSTM), 2, "r");
//
//         /// <summary>
//         /// Gets b.
//         /// </summary>
//         public static readonly ParameterInfo B = new(typeof(LSTM), 3, "b");
//
//         /// <summary>
//         /// Gets initial_h.
//         /// </summary>
//         public static readonly ParameterInfo initial_h = new(typeof(LSTM), 4, "initial_h",
//             HasDataType(DataTypes.Float32));
//
//         /// <summary>
//         /// Gets initial_c.
//         /// </summary>
//         public static readonly ParameterInfo initial_c = new(typeof(LSTM), 5, "initial_c",
//             HasDataType(DataTypes.Float32));
//
//         /// <summary>
//         /// Gets has_static.
//         /// </summary>
//         public static readonly ParameterInfo has_static = new(typeof(LSTM), 6, "has_static", IsBool());
//     }
// }
