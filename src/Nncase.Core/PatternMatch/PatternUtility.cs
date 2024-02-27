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
    /// <summary>
    /// call binary swap.
    /// </summary>
    /// <param name="targetName">target name.</param>
    /// <param name="callName">call name. </param>
    /// <param name="condition">binary op condition.</param>
    /// <param name="lhs">lhs pattern.</param>
    /// <param name="rhs">rhs pattern.</param>
    /// <returns>pattern.</returns>
    public static Pattern IsSwappableBinary(string targetName, string? callName, Func<Binary, bool> condition, Pattern lhs, Pattern rhs) =>
      IsAlt(
        IsBinary(targetName, callName, condition, lhs, rhs),
        IsBinary(targetName, callName, condition, rhs, lhs));

    /// <summary>
    /// Generate VArgsPattern with name = "pre_fix"+"Params".
    /// </summary>
    /// <param name="prefix">prefix.</param>
    /// <param name="patterns">input patterns.</param>
    /// <returns>VArgsPattern.</returns>
    public static VArgsPattern GenerateParameters(string? prefix, params Pattern[] patterns)
      => IsVArgsRepeat(
          (prefix is not null && prefix != string.Empty) ? prefix + "Params" : null,
          list =>
          {
              return patterns.Concat(
                      Enumerable.Range(0, Math.Max(list.Length - patterns.Length, 0)).Select(_ => (Pattern)IsWildcard(null)))
                  .ToArray();
          });

    /// <summary>
    /// generate postion specific vargs pattern.
    /// </summary>
    /// <param name="prefix">prefix.</param>
    /// <param name="inputPatterns">info with pattern pairs.</param>
    /// <returns>VArgsPattern.</returns>
    public static VArgsPattern GenerateParameters(string prefix, (ParameterInfo Info, Pattern Pattern)[] inputPatterns)
      => IsVArgsRepeat(
          (prefix is not null && prefix != string.Empty) ? prefix + "Params" : null,
          list =>
          {
              var patterns = Enumerable.Range(0, list.Length).Select(_ => (Pattern)IsWildcard()).ToArray();
              foreach (var (info, pattern) in inputPatterns)
              {
                  patterns[info.Index] = pattern;
              }

              return patterns;
          });

    /// <summary>
    /// is call wildcard inputs.
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

    /// <summary>
    /// call wildcard and swap it.
    /// </summary>
    /// <param name="callName">call name.</param>
    /// <param name="targetPattern">target pattern.</param>
    /// <param name="lhsPattern">lhs pattern.</param>
    /// <param name="rhsPattern">rhs pattern.</param>
    /// <returns>pattern.</returns>
    public static Pattern IsCallWildcardSwappable(string callName, Pattern targetPattern, Pattern lhsPattern, Pattern rhsPattern) =>
        IsAlt(
            IsCallWildcard(callName, targetPattern, lhsPattern, rhsPattern),
            IsCallWildcard(callName, targetPattern, rhsPattern, lhsPattern));

    /// <summary>
    /// wrapped lstm pattern.
    /// </summary>
    /// <param name="lstmPattern">lstm pattern.</param>
    /// <param name="wrapFunc"> wrap pattern func.</param>
    /// <returns>pattern.</returns>
    public static Pattern IsWrappedLSTM(CallPattern lstmPattern, Func<Pattern, int, Pattern> wrapFunc) =>
        IsTuple("tuple", IsVArgsRepeat("xx", tp =>
        {
            var patterns = new Pattern[tp.Length];
            for (var i = 0; i < tp.Length; i++)
            {
                patterns[i] = wrapFunc(F.Tensors.IsGetItem(lstmPattern, IsWildcard()), i);
            }

            return patterns;
        }));

    /// <summary>
    /// Call pattern with spec multi input pattern.
    /// </summary>
    /// <param name="callName">call name.</param>
    /// <param name="targetPattern">call target pattern.</param>
    /// <param name="inputPatterns"> postions pattern pairs.</param>
    /// <returns>call pattern.</returns>
    public static CallPattern IsCallSpecific(string callName, Pattern targetPattern, params (ParameterInfo Info, Pattern Pattern)[] inputPatterns)
        => IsCall(callName, targetPattern, GenerateParameters(callName, inputPatterns));

    /// <summary>
    /// single input pattern body pattern.
    /// </summary>
    /// <typeparam name="T">mid op type.</typeparam>
    /// <typeparam name="TBegin">begin op type.</typeparam>
    /// <typeparam name="TEnd">end op type.</typeparam>
    /// <param name="endName">end name.</param>
    /// <param name="midName">min name.</param>
    /// <param name="beginName">begin name.</param>
    /// <param name="inputName">input name.</param>
    /// <returns>single input pattern.</returns>
    public static Pattern IsSIFusionBody<T, TBegin, TEnd>(string endName, string midName, string beginName, string inputName)
        where T : Op
        where TBegin : Op
        where TEnd : Op =>
      IsCallWildcard(
        endName,
        IsOp<TEnd>(endName + "Op"),
        IsCallWildcard(
          midName,
          IsOp<T>(midName + "Op"),
          IsCallWildcard(
            beginName,
            IsOp<TBegin>(beginName + "Op"),
            IsWildcard(inputName))));

    /// <summary>
    /// is double input fusion body.
    /// </summary>
    /// <typeparam name="T">mid op type.</typeparam>
    /// <typeparam name="TBegin">begin op type.</typeparam>
    /// <typeparam name="TEnd">end op type.</typeparam>
    /// <param name="endName">end name.</param>
    /// <param name="midName">min name.</param>
    /// <param name="beginName">begin name.</param>
    /// <param name="inputName">input name.</param>
    /// <returns>pattern.</returns>
    public static Pattern IsDIFusionBody<T, TBegin, TEnd>(string endName, string midName, string beginName, string inputName)
        where T : Op
        where TBegin : Op
        where TEnd : Op =>
      IsCallWildcard(
        endName,
        IsOp<TEnd>(endName + "Op"),
        IsCallWildcard(
          midName,
          IsOp<T>(midName + "Op"),
          IsCallWildcard(beginName + "Lhs", IsOp<TBegin>(beginName + "LhsOp"), IsWildcard(inputName + "Lhs")),
          IsCallWildcard(beginName + "Rhs", IsOp<TBegin>(beginName + "RhsOp"), IsWildcard(inputName + "Rhs"))));

    /// <summary>
    /// is double input fusion body.
    /// </summary>
    /// <typeparam name="T">mid op type.</typeparam>
    /// <typeparam name="TBegin">begin op type.</typeparam>
    /// <typeparam name="TEnd">end op type.</typeparam>
    /// <param name="fusionName">fusion name.</param>
    /// <param name="moduleKind">module kind.</param>
    /// <param name="endName">end prefix.</param>
    /// <param name="midName"> mid prefix.</param>
    /// <param name="beginName">begin prefix.</param>
    /// <param name="inputName">input name.</param>
    /// <returns>pattern.</returns>
    public static Pattern IsFusion<T, TBegin, TEnd>(string fusionName, string moduleKind, string endName, string midName, string beginName, string inputName)
        where T : Op
        where TBegin : Op
        where TEnd : Op =>
      IsFusion(
        fusionName,
        moduleKind,
        IsAlt(
          IsSIFusionBody<T, TBegin, TEnd>(endName, midName, beginName, inputName),
          IsDIFusionBody<T, TBegin, TEnd>(endName, midName, beginName, inputName)),
        IsVArgsRepeat(() => IsVar()));

    /// <summary>
    /// is any fusion.
    /// </summary>
    /// <param name="moduleKind">module kind.</param>
    /// <param name="body">fusion body pattern.</param>
    /// <returns>pattern.</returns>
    public static Pattern IsFusion(string moduleKind, Pattern body)
        => IsFusion("fusion", moduleKind, body, IsVArgsRepeat(null, () => IsVar()));

    /// <summary>
    /// Fusion: TBegin -> FirstOp -> SecondOp -> TEnd.
    /// </summary>
    /// <param name="patternCtorA">ctor for a.</param>
    /// <param name="patternCtorB">ctor for b.</param>
    /// <returns>ctor.</returns>
    public static PatternCtor IsAlt(PatternCtor patternCtorA, PatternCtor patternCtorB) => input =>
        IsAlt(patternCtorA(input), patternCtorB(input));

    /// <summary>
    /// paired wildcared call.
    /// </summary>
    /// <typeparam name="TFirstOp">first call op type.</typeparam>
    /// <typeparam name="TSecondOp">second call op type.</typeparam>
    /// <param name="firstCallName">first call op name.</param>
    /// <param name="secondCallName">second call op name.</param>
    /// <param name="input">input pattern.</param>
    /// <returns>pattern.</returns>
    public static Pattern IsPairWildcardCall<TFirstOp, TSecondOp>(string firstCallName, string secondCallName, Pattern input)
        where TFirstOp : Op
        where TSecondOp : Op
        => IsCallWildcardMaybeSwappable<TSecondOp>(secondCallName, IsCallWildcardMaybeSwappable<TFirstOp>(firstCallName, input));

    /// <summary>
    /// pair fusion with the begin end.
    /// </summary>
    /// <typeparam name="TFirstOp">first call op type.</typeparam>
    /// <typeparam name="TSecondOp">second call op type.</typeparam>
    /// <typeparam name="TBegin">Begin name.</typeparam>
    /// <typeparam name="TEnd">End name.</typeparam>
    /// <param name="moduleKind">kind.</param>
    /// <param name="firstCallName">name.</param>
    /// <returns>pattern.</returns>
    public static Pattern IsPairLayerFusion<TFirstOp, TSecondOp, TBegin, TEnd>(string moduleKind, string firstCallName)
        where TFirstOp : Op
        where TSecondOp : Op
        where TBegin : Op
        where TEnd : Op =>
        IsPairLayerFusion<TFirstOp, TSecondOp, TBegin, TEnd>(moduleKind, "ld", firstCallName, "st");

    /// <summary>
    /// get pair layer fusion with name.
    /// </summary>
    /// <typeparam name="TFirstOp">first call op type.</typeparam>
    /// <typeparam name="TSecondOp">second call op type.</typeparam>
    /// <typeparam name="TBegin">Begin name.</typeparam>
    /// <typeparam name="TEnd">End name.</typeparam>
    /// <param name="moduleKind">kind.</param>
    /// <param name="beginCallName">begin name.</param>
    /// <param name="firstCallName">name.</param>
    /// <param name="endCallName">end name.</param>
    /// <returns>pair pattern.</returns>
    public static Pattern IsPairLayerFusion<TFirstOp, TSecondOp, TBegin, TEnd>(string moduleKind, string beginCallName, string firstCallName, string endCallName)
        where TFirstOp : Op
        where TSecondOp : Op
        where TBegin : Op
        where TEnd : Op =>
      IsFusion(
        moduleKind,
        IsCallWildcard(
          endCallName,
          IsOp<TEnd>(endCallName + "Op"),
          IsAlt(
            input => IsPairWildcardCall<TFirstOp, TSecondOp>(firstCallName, null!, input),
            input => IsCallWildcardMaybeSwappable<TFirstOp>(firstCallName, input))(
              IsCallWildcard(
                beginCallName,
                IsOp<TBegin>(beginCallName + "Op"),
                IsWildcard()))));

    /// <summary>
    /// get call wildcard maybe swappable.
    /// </summary>
    /// <typeparam name="TOp">op type.</typeparam>
    /// <param name="callName">call prefix.</param>
    /// <param name="input">input name.</param>
    /// <param name="swappableOther">the other pattern.</param>
    /// <returns>pattern.</returns>
    public static Pattern IsCallWildcardMaybeSwappable<TOp>(string callName, Pattern input, Pattern? swappableOther = null)
        where TOp : Op =>
      IsAlt(
        IsCallWildcard(callName, IsOp<TOp>(callName + "Op"), input),
        IsCallWildcardSwappable(callName, IsOp<TOp>(callName + "Op"), input, swappableOther ?? IsWildcard()));

    public static Pattern MaybeMarker(Pattern input) => IsAlt(input, IsRangeOfMarker(input, IsWildcard()));

    public static Pattern MaybeMarker(Pattern input, string markerName) => IsAlt(input, IsRangeOfMarker(markerName, input, IsWildcard()));

    public static Pattern HasMarker(Pattern input, string? markerName = null) => IsRangeOfMarker(markerName, input, IsWildcard());
}
