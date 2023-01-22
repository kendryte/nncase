// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System.Reflection;
using Nncase.IR;
using Nncase.IR.Math;
using static Nncase.IR.TypePatternUtility;
using static Nncase.PatternMatch.F.Math;
using static Nncase.Utilities.ReplaceUtility;
using ParameterInfo = Nncase.IR.ParameterInfo;
using PatternCtor = System.Func<Nncase.PatternMatch.Pattern, Nncase.PatternMatch.Pattern>;

namespace Nncase.PatternMatch;

public static partial class Utility
{
    // public static VArgsPattern WildcardVArgsPattern => GenerateRepeatParameters(IsWildcard);

    public static Pattern
        IsSwappableBinary(string targetName, Func<Binary, bool> condition, Pattern lhs, Pattern rhs)
        => IsSwappableBinary(targetName, null, condition, lhs, rhs);

    public static Pattern
        IsSwappableBinary(string targetName, string? callName, Func<Binary, bool> condition, Pattern lhs,
            Pattern rhs) => IsAlt(
        IsBinary(targetName, callName, condition, lhs, rhs),
        IsBinary(targetName, callName, condition, rhs, lhs));

    /// <summary>
    /// Generate VArgsPattern with name = "pre_fix"+"Params".
    /// </summary>
    /// <param name="prefix">prefix.</param>
    /// <param name="patterns">input patterns.</param>
    /// <returns></returns>
    public static VArgsPattern GenerateParameters<T>(string? prefix, params T[] patterns)
      where T : Pattern =>
        IsVArgsRepeat((prefix is not null && prefix != string.Empty) ? prefix + "Params" : null,
          list => patterns.Concat(
            Enumerable.Range(0, list.Count - patterns.Length).
            Select(_ => (Pattern)IsWildcard(null))
          ).ToArray());

    /// <summary>
    /// generate postion specific vargs pattern.
    /// </summary>
    /// <param name="prefix">prefix.</param>
    /// <param name="inputPatterns"></param>
    /// <returns></returns>
    /// <exception cref="NotImplementedException"></exception>
    public static VArgsPattern GenerateParameters<T>(string prefix, (ParameterInfo info, T pattern)[] inputPatterns)
      where T : Pattern =>
        IsVArgsRepeat((prefix is not null && prefix != string.Empty) ? prefix + "Params" : null,
          list =>
          {
              var patterns = Enumerable.Range(0, list.Count).Select(_ => (Pattern)IsWildcard()).ToArray();
              foreach (var (info, pattern) in inputPatterns)
              {
                  patterns[info.Index] = pattern;
              }
              return patterns;
          }
        );

    /// <summary>
    /// is call wildcard inputs
    /// <remarks>
    /// will generate callName+Params for matched inputs.
    /// </remarks>
    /// </summary>
    /// <param name="callName">call name.</param>
    /// <param name="targetPattern">target pattern.</param>
    /// <param name="patterns">patterns by sequence.</param>
    /// <returns>call pattern. </returns>
    public static CallPattern IsCallWildcard(string? callName, Pattern targetPattern, params Pattern[] patterns) =>
        IsCall(callName, targetPattern, GenerateParameters(callName, patterns));

    public static Pattern IsCallWildcardSwappable<T>(string callName, string opName, Pattern lhsPattern, Pattern rhsPattern)
      where T : Op =>
        IsAlt(IsCallWildcard(callName, IsOp<T>(opName), lhsPattern, rhsPattern),
              IsCallWildcard(callName, IsOp<T>(opName), rhsPattern, lhsPattern));

    // public static CallPattern IsWildcardCall(string callName, Pattern firstInputPattern) =>
    //     IsCall(callName, IsWildcard(), GenerateParameters(callName, firstInputPattern))
    //         with
    //     {
    //         TypePattern = IsType(x => !(x is InvalidType)),
    //     };

    // public static VArgsPattern GenerateRepeatParameters(Func<Pattern> pGenerator) =>
    //     IsVArgsRepeat(list =>
    //         Enumerable.Range(0, list.Count).Select(_ => pGenerator())
    //             .ToArray());

    // public static VArgsPattern ParamsWithArg(Pattern argPattern) => new VArgsPattern(
    //     (fields) =>
    //     {
    //         var fieldList = fields.ToList();
    //         var i = fieldList.FindIndex(f => CompilerServices.TryMatch(f, argPattern, out var s));
    //         return i == -1

    //             // force match failed
    //             ? Enumerable.Repeat(IsWildcard(), fields.Count + 1).ToArray()
    //             : ReplacePos(fields.Select(_ => IsWildcard()).ToArray(), argPattern, i);
    //     }, null);

    /// <summary>
    /// GenerateParameters for spec multi param.
    /// </summary>
    /// <param name="prefix"></param>
    /// <param name="specPattern"></param>
    /// <returns></returns>
    // public static VArgsPattern GenerateParameters(string prefix, (ParameterInfo, Pattern)[] specPattern) =>
    //     IsVArgsRepeat(
    //         prefix != null ? prefix + "Params" : null!,
    //         list =>
    //             ReplaceMulti(
    //                 Enumerable
    //                     .Range(0, list.Count)
    //                     .Select(_ => IsWildcard(null))
    //                     .ToArray(),
    //                 specPattern));

    /// <summary>
    /// Call pattern with spec multi input pattern.
    /// </summary>
    /// <param name="callName"></param>
    /// <param name="targetPattern"></param>
    /// <param name="inputPatterns"> postions pattern pairs</param>
    /// <returns></returns>
    public static CallPattern IsCallSpecific<T>(string callName, Pattern targetPattern,
        params (ParameterInfo info, T pattern)[] inputPatterns)
          where T : Pattern
        => IsCall(callName, targetPattern, GenerateParameters<T>(callName, inputPatterns));

    // public static int Count<T>()
    //     where T : Op
    //     => typeof(T).GetFields(BindingFlags.Public | BindingFlags.Static).Length;

    ///
    /// <returns></returns><summary>
    /// generate a vargs pattern everything is wildcard except for the specified index
    /// e.g.
    /// ArgsPattern.<GNNEConv2D>(
    ///     (GNNEConv2D.Weights, IsTensorConst()),
    ///     (GNNEConv2D.PSum, IsNone())
    /// )
    /// </summary>
    /// <param name="specs"></param>
    /// <typeparam name="T"></typeparam>
    /// <returns></returns>
    // public static VArgsPattern ArgsPattern<T>(params (ParameterInfo, Pattern)[] specs)
    //     where T : Op
    // {
    //     // todo:完善文档
    //     var wildcards = FieldsPatternGenerator(IsWildcard, Count<T>());
    //     var fields = ReplaceMulti(wildcards, specs);
    //     return new VArgsPattern(fields, null);
    // }

    // public static CallPattern IsCallWithSpec<T>(string callName, string opName, params (ParameterInfo, Pattern)[] specs)
    //     where T : Op =>
    //     IsCall(callName, IsOp<T>(opName, _ => true), GenerateParameters(callName))

    /// <summary>
    /// e.g.
    /// deq  deq  deq
    ///  \    |    /
    ///    concat.
    /// </summary>
    /// <param name="p"></param>
    /// <param name="name"></param>
    /// <returns></returns>
    // public static Pattern IsRepeatTuple(Func<Pattern> pGenerator, string? name = null) => IsTuple(
    //     new VArgsPattern(
    //         fields => FieldsPatternGenerator(pGenerator, fields.Count), null),
    //     name);

    // public static Pattern[] FieldsPatternGenerator(Func<Pattern> pGenerator, int count) =>
    //     Enumerable.Range(0, count).Select(_ => pGenerator()).ToArray();

    // todo: replace pattern in SingleInputFusion and DoubleInputFusion with IsSIFusionBody and IsDIFusionBody

    /// <summary>
    /// is single input body.
    /// </summary>
    /// <typeparam name="T"></typeparam>
    /// <typeparam name="BeginT"></typeparam>
    /// <typeparam name="EndT"></typeparam>
    /// <returns></returns>
    public static Pattern IsSIFusionBody<T, BeginT, EndT>(string mid_name, string inputName = "input",
        string callName = "call", string beginName = "ld", string endName = "st")
        where T : Op
        where BeginT : Op
        where EndT : Op => IsCallWildcard(endName, IsOp<EndT>(endName + "Op"),
        IsCallWildcard(callName, IsOp<T>(mid_name + "Op"),
            IsCallWildcard(beginName, IsOp<BeginT>(beginName + "Op"), IsWildcard(inputName))));

    /// <summary>
    /// is double input fusion body.
    /// </summary>
    /// <typeparam name="T"></typeparam>
    /// <typeparam name="T"></typeparam>
    /// <typeparam name="BeginT"></typeparam>
    /// <typeparam name="EndT"></typeparam>
    /// <returns></returns>
    // public static Pattern IsDIFusionBody<T, BeginT, EndT>(string callName = "call")
    //     where T : Op
    //     where BeginT : Op
    //     where EndT : Op => IsCallWildcard<EndT>("st", null!,
    //     IsCallWildcard<T>(callName, null!,
    //         IsCallWildcard<BeginT>(null!, null!, IsWildcard("lhs")),
    //         IsCallWildcard<BeginT>(null!, null!, IsWildcard("rhs"))));

    // public static Pattern IsFusion<T, BeginT, EndT>(string mid_name, string module_kind, string inputName = "input",
    //     string callName = "call", string beginName = "ld", string endName = "st", string fusionName = "fusion")
    //     where T : Op
    //     where BeginT : Op
    //     where EndT : Op => IsFusion(fusionName, module_kind,
    //     IsAlt(
    //         IsSIFusionBody<T, BeginT, EndT>(mid_name, inputName, callName, beginName, endName),
    //         IsDIFusionBody<T, BeginT, EndT>(callName)),
    //     WildcardVArgsPattern);

    public static Pattern IsFusion(string module_kind, Pattern body)
        => IsFusion(null, module_kind, body,
            IsVArgsRepeat(null, () => IsVar()));

    // Fusion: BeginT -> FirstOp -> SecondOp -> EndT
    public static PatternCtor IsAlt(PatternCtor patternCtorA, PatternCtor patternCtorB) => input =>
        IsAlt(patternCtorA(input), patternCtorB(input));

    public static Pattern IsPairWildcardCall<FirstOpT, SecondOpT>(string firstCallName, string secondCallName, Pattern input)
        where FirstOpT : Op
        where SecondOpT : Op
        => IsCallWildcardMaybeSwappable<SecondOpT>(secondCallName, IsCallWildcardMaybeSwappable<FirstOpT>(firstCallName, input));

    public static Pattern IsPairLayerFusion<FirstOpT, SecondOpT, BeginT, EndT>(string moduleKind, string firstCallName)
        where FirstOpT : Op
        where SecondOpT : Op
        where BeginT : Op
        where EndT : Op =>
        IsPairLayerFusion<FirstOpT, SecondOpT, BeginT, EndT>(moduleKind, "ld", firstCallName, "st");

    public static Pattern IsPairLayerFusion<FirstOpT, SecondOpT, BeginT, EndT>(string moduleKind, string beginCallName, string firstCallName, string endCallName)
        where FirstOpT : Op
        where SecondOpT : Op
        where BeginT : Op
        where EndT : Op => IsFusion(
        moduleKind,
        IsCallWildcard(endCallName, IsOp<EndT>(endCallName + "Op"),
            IsAlt( // we can't use secondCallName in getReplace because of it's optional
                input => IsPairWildcardCall<FirstOpT, SecondOpT>(firstCallName, null!, input),
                input => IsCallWildcardMaybeSwappable<FirstOpT>(firstCallName, input))(
                IsCallWildcard(beginCallName, IsOp<BeginT>(beginCallName + "Op"), IsWildcard()))));

    public static Pattern IsCallWildcardMaybeSwappable<OpT>(string callName, Pattern input)
        where OpT : Op =>
        IsCallWildcardMaybeSwappable<OpT>(callName, input, IsWildcard());

    public static Pattern IsCallWildcardMaybeSwappable<OpT>(string callName, Pattern input, Pattern swappableOther)
        where OpT : Op => IsAlt(
        IsCallWildcard(callName, IsOp<OpT>(callName + "Op"), input),
        IsCallWildcardSwappable<OpT>(callName, null!, input, swappableOther));
}
