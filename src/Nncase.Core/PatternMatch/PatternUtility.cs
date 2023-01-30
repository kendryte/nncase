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
    public static VArgsPattern WildcardVArgsPattern => GenerateRepeatParameters(IsWildcard);

    public static Pattern
        IsSwappableBinary(string targetName, Func<Binary, bool> condition, Pattern lhs, Pattern rhs)
        => IsSwappableBinary(targetName, null, condition, lhs, rhs);

    public static Pattern
        IsSwappableBinary(string targetName, string? callName, Func<Binary, bool> condition, Pattern lhs, Pattern rhs) => IsAlt(
        IsBinary(targetName, callName, condition, lhs, rhs),
        IsBinary(targetName, callName, condition, rhs, lhs));

    /// <summary>
    /// Generate VArgsPattern with name = "pre_fix"+"Params".
    /// </summary>
    /// <param name="prefix">prefix.</param>
    /// <param name="beginPatterns">input pattern.</param>
    public static VArgsPattern GenerateParameters(string? prefix, Pattern[] beginPatterns) =>
        IsVArgsRepeat(
            prefix != null ? prefix + "Params" : null!,
            list =>
                beginPatterns
                    .Concat(Enumerable.Range(0, list.Count - beginPatterns.Length).Select(_ => IsWildcard(null)))
                    .ToArray());

    /// <summary>
    /// Generate VArgsPattern with name = "pre_fix"+"Params".
    /// </summary>
    /// <param name="prefix">prefix.</param>
    /// <param name="inputPattern">input pattern.</param>
    public static VArgsPattern GenerateParameters(string? prefix, Pattern inputPattern) =>
        GenerateParameters(prefix, new[] { inputPattern });

    /// <summary>
    /// match a call with op type T
    /// auto set first param
    /// it's always used for Fake to NoFake Pass with ReplaceCall.
    /// </summary>
    /// <typeparam name="T">Op Type.</typeparam>
    public static CallPattern IsWildcardCall<T>(string? callName, string? opName, string? inputName = "input")
        where T : Op =>
        IsWildcardCall<T>(callName, opName, IsWildcard(inputName));

    public static CallPattern IsWildcardCall<T>(string? callName, string? opName, Pattern inputPattern)
        where T : Op
        =>
            IsCall(callName, IsOp<T>(opName, _ => true), GenerateParameters(callName, inputPattern))
                with
            {
                TypePattern = IsType(x => !(x is InvalidType)),
            };

    public static CallPattern IsWildcardCall<T>(string? callName, string? opName, Pattern lhsPattern, Pattern rhsPattern)
        where T : Op =>
        IsCall(callName, IsOp<T>(opName, _ => true), GenerateParameters(callName, new[] { lhsPattern, rhsPattern }))
            with
        {
            TypePattern = IsType(x => !(x is InvalidType)),
        };

    public static Pattern IsSwappableWildcardCall<T>(string? callName, string? opName, Pattern lhsPattern, Pattern rhsPattern)
        where T : Op
        =>
            IsAlt(
                IsWildcardCall<T>(callName, opName, lhsPattern, rhsPattern),
                IsWildcardCall<T>(callName, opName, rhsPattern, lhsPattern));

    public static CallPattern IsWildcardCall(string? callName, Pattern firstInputPattern) =>
        IsCall(callName, IsWildcard(), GenerateParameters(callName, firstInputPattern))
            with
        {
            TypePattern = IsType(x => !(x is InvalidType)),
        };

    public static VArgsPattern GenerateRepeatParameters(Func<Pattern> pGenerator) =>
        IsVArgsRepeat(list =>
            Enumerable.Range(0, list.Count).Select(_ => pGenerator())
                .ToArray());

    public static VArgsPattern ParamsWithArg(Pattern argPattern) => new VArgsPattern(
        (fields) =>
        {
            var fieldList = fields.ToList();
            var i = fieldList.FindIndex(f => CompilerServices.TryMatch(f, argPattern, out var s));
            return i == -1

                // force match failed
                ? Enumerable.Repeat(IsWildcard(), fields.Count + 1).ToArray()
                : ReplacePos(fields.Select(_ => IsWildcard()).ToArray(), argPattern, i);
        },
        null);

    public static CallPattern IsWildcardCall<T>(string? opName = null!)
        where T : Op
        => IsWildcardCall<T>("call", opName);

    /// <summary>
    /// GenerateParameters for spec multi param.
    /// </summary>
    public static VArgsPattern GenerateParameters(string? prefix, (ParameterInfo, Pattern)[] specPattern) =>
        IsVArgsRepeat(
            prefix != null ? prefix + "Params" : null!,
            list =>
                ReplaceMulti(
                    Enumerable
                        .Range(0, list.Count)
                        .Select(_ => IsWildcard(null))
                        .ToArray(),
                    specPattern));

    /// <summary>
    /// Call pattern with spec multi input pattern.
    /// </summary>
    public static CallPattern IsCallWithSpecInput<T>(string? callName, string? opName, (ParameterInfo, Pattern)[] inputPattern)
        where T : Op => IsCall(callName, IsOp<T>(opName, _ => true), GenerateParameters(callName, inputPattern))
        with
        {
            TypePattern = IsType(x => !(x is InvalidType)),
        };

    public static int Count<T>()
        where T : Op
        => typeof(T).GetFields(BindingFlags.Public | BindingFlags.Static).Length;

    /// <summary>
    /// generate a vargs pattern everything is wildcard except for the specified index
    /// e.g.
    /// ArgsPattern.&lt;GNNEConv2D&gt;(
    ///     (GNNEConv2D.Weights, IsTensorConst()),
    ///     (GNNEConv2D.PSum, IsNone())
    /// ).
    /// </summary>
    public static VArgsPattern ArgsPattern<T>(params (ParameterInfo, Pattern)[] specs)
        where T : Op
    {
        // todo:完善文档
        var wildcards = FieldsPatternGenerator(IsWildcard, Count<T>());
        var fields = ReplaceMulti(wildcards, specs);
        return new VArgsPattern(fields, null);
    }

    public static CallPattern IsCallWithSpec<T>(string callName, string opName, params (ParameterInfo, Pattern)[] specs)
        where T : Op =>
        IsCall(callName, IsOp<T>(opName, _ => true), ArgsPattern<T>(specs))
            with
        {
            TypePattern = IsType(x => !(x is InvalidType)),
        };

    /// <summary>
    /// e.g.
    /// deq  deq  deq
    ///  \    |    /
    ///    concat.
    /// </summary>
    public static Pattern IsRepeatTuple(Func<Pattern> pGenerator, string? name = null) => IsTuple(
        new VArgsPattern(
            fields => FieldsPatternGenerator(pGenerator, fields.Count), null),
        name);

    public static Pattern[] FieldsPatternGenerator(Func<Pattern> pGenerator, int count) =>
        Enumerable.Range(0, count).Select(_ => pGenerator()).ToArray();

    // todo: replace pattern in SingleInputFusion and DoubleInputFusion with IsSIFusionBody and IsDIFusionBody

    /// <summary>
    /// is single input body.
    /// </summary>
    public static Pattern IsSIFusionBody<T, TBegin, TEnd>(string mid_name, string inputName = "input", string callName = "call", string beginName = "ld", string endName = "st")
        where T : Op
        where TBegin : Op
        where TEnd : Op => IsWildcardCall<TEnd>(
            endName,
            null!,
            IsWildcardCall<T>(callName, mid_name, IsWildcardCall<TBegin>(beginName, null!, IsWildcard(inputName))));

    /// <summary>
    /// is double input fusion body.
    /// </summary>
    public static Pattern IsDIFusionBody<T, TBegin, TEnd>(string callName = "call")
        where T : Op
        where TBegin : Op
        where TEnd : Op => IsWildcardCall<TEnd>(
            "st",
            null!,
            IsWildcardCall<T>(
                callName,
                null!,
                IsWildcardCall<TBegin>(null!, null!, IsWildcard("lhs")),
                IsWildcardCall<TBegin>(null!, null!, IsWildcard("rhs"))));

    public static Pattern IsFusion<T, TBegin, TEnd>(string mid_name, string module_kind, string inputName = "input", string callName = "call", string beginName = "ld", string endName = "st", string fusionName = "fusion")
        where T : Op
        where TBegin : Op
        where TEnd : Op => IsFusion(
            fusionName,
            module_kind,
            IsAlt(
                IsSIFusionBody<T, TBegin, TEnd>(mid_name, inputName, callName, beginName, endName),
                IsDIFusionBody<T, TBegin, TEnd>(callName)),
            WildcardVArgsPattern);

    public static Pattern IsFusion(string module_kind, Pattern body)
        => IsFusion(null, module_kind, body, IsVArgsRepeat(null, () => IsVar()));

    // Fusion: TBegin -> FirstOp -> SecondOp -> TEnd
    public static PatternCtor IsAlt(PatternCtor patternCtorA, PatternCtor patternCtorB) => input =>
        IsAlt(patternCtorA(input), patternCtorB(input));

    public static Pattern IsPairWildcardCall<TFirstOp, TSecondOp>(string firstCallName, string secondCallName, Pattern input)
        where TFirstOp : Op
        where TSecondOp : Op => IsMaybeSwappableWildcardCall<TSecondOp>(
        secondCallName,
        IsMaybeSwappableWildcardCall<TFirstOp>(
            firstCallName, input));

    public static Pattern IsPairLayerFusion<TFirstOp, TSecondOp, TBegin, TEnd>(
        string moduleKind,
        string firstCallName)
        where TFirstOp : Op
        where TSecondOp : Op
        where TBegin : Op
        where TEnd : Op =>
        IsPairLayerFusion<TFirstOp, TSecondOp, TBegin, TEnd>(moduleKind, "ld", firstCallName, "st");

    public static Pattern IsPairLayerFusion<TFirstOp, TSecondOp, TBegin, TEnd>(string moduleKind, string beginCallName, string firstCallName, string endCallName)
        where TFirstOp : Op
        where TSecondOp : Op
        where TBegin : Op
        where TEnd : Op => IsFusion(
        moduleKind,
        IsWildcardCall<TEnd>(
            endCallName,
            null!,
            /* we can't use secondCallName in getReplace because of it's optional */
            IsAlt(
                input => IsPairWildcardCall<TFirstOp, TSecondOp>(firstCallName, null!, input),
                input => IsMaybeSwappableWildcardCall<TFirstOp>(firstCallName, input))(
                IsWildcardCall<TBegin>(beginCallName, null!, (string)null!))));

    public static Pattern IsMaybeSwappableWildcardCall<TOp>(string callName, Pattern input)
        where TOp : Op => IsMaybeSwappableWildcardCall<TOp>(callName, input, IsWildcard());

    public static Pattern IsMaybeSwappableWildcardCall<TOp>(string callName, Pattern input, Pattern swappableOther)
        where TOp : Op => IsAlt(
        IsWildcardCall<TOp>(callName, null!, input),
        IsSwappableWildcardCall<TOp>(callName, null!, input, swappableOther));
}
