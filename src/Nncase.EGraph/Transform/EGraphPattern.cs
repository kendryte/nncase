// // Copyright (c) Canaan Inc. All rights reserved.
// // Licensed under the Apache license. See LICENSE file in the project root for full license information.

// using System;
// using System.Collections.Generic;
// using System.Linq;
// using Nncase.IR;

// namespace Nncase.Transform.EGraphPatterns
// {

//     public delegate bool Condition<in T>(T obj);

//     /// <summary>
//     /// EGraph pattern.
//     /// </summary>
//     public abstract record EGraphPattern
//     {
//     }

//     /// <summary>
//     /// Wildcard pattern.
//     /// </summary>
//     public sealed record WildcardPattern() : EGraphPattern;

//     /// <summary>
//     /// Variable pattern.
//     /// </summary>
//     public sealed record VarPattern(Var expr, Condition<Var> cond) : EGraphPattern
//     {
//     }

//     public sealed record ConstPattern(Const? expr, Condition<Const> cond) : EGraphPattern
//     {

//     }

//     /// <summary>
//     /// Functional patterns.
//     /// </summary>
    
// }
