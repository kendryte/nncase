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
        IsSwappableBinary(string targetName, string? callName, Func<Binary, bool> condition, Pattern lhs,
            Pattern rhs) => IsAlt(
        IsBinary(targetName, callName, condition, lhs, rhs),
        IsBinary(targetName, callName, condition, rhs, lhs));

    public static VArgsPattern GenerateParameters(Pattern inputPattern) =>
        GenerateParameters(new[] { inputPattern });

    private static VArgsPattern GenerateParameters(Pattern[] beginPatterns) =>
        IsVArgsRepeat(list =>
            beginPatterns
                .Concat(Enumerable.Range(0, list.Count - beginPatterns.Length).Select(_ => IsWildcard(null)))
                .ToArray());

    /// <summary>
    /// match a call with op type T
    /// auto set first param
    /// it's always used for Fake to NoFake Rule with ReplaceCall.
    /// </summary>ReplaceParams
    /// <param name="callName"></param>
    /// <param name="opName"></param>
    /// <typeparam name="T">Op Type.</typeparam>
    /// <returns></returns>
    public static CallPattern IsWildcardCall<T>(string callName, string opName, string inputName = "input")
        where T : Op =>
        IsWildcardCall<T>(callName, opName, IsWildcard(inputName));

    public static CallPattern IsWildcardCall<T>(string callName, string opName, Pattern inputPattern)
        where T : Op
        =>
        IsCall(callName, IsOp<T>(opName, _ => true), GenerateParameters(inputPattern))
            with
        {
            TypePattern = IsType(x => !(x is InvalidType)),
        };

    public static CallPattern IsWildcardCall<T>(string callName, string opName, Pattern lhsPattern, Pattern rhsPattern)
        where T : Op =>
        IsCall(callName, IsOp<T>(opName, _ => true), GenerateParameters(new[] { lhsPattern, rhsPattern }))
            with
        {
            TypePattern = IsType(x => !(x is InvalidType)),
        };

    public static Pattern IsSwappableWildcardCall<T>(string callName, string opName, Pattern lhsPattern,
        Pattern rhsPattern)
        where T : Op
        =>
        IsAlt(
            IsWildcardCall<T>(callName, opName, lhsPattern, rhsPattern),
            IsWildcardCall<T>(callName, opName, rhsPattern, lhsPattern));

    public static CallPattern IsWildcardCall(string callName, Pattern firstInputPattern) =>
        IsCall(callName, IsWildcard(), GenerateParameters(firstInputPattern))
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
    }, null);

    public static CallPattern IsWildcardCall<T>(string opName = null!)
        where T : Op
        =>
        IsWildcardCall<T>("call", opName);

    public static int Count<T>()
        where T : Op
        => typeof(T).GetFields(BindingFlags.Public | BindingFlags.Static).Length;

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
    /// <param name="p"></param>
    /// <param name="name"></param>
    /// <returns></returns>
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
    /// <typeparam name="T"></typeparam>
    /// <typeparam name="BeginT"></typeparam>
    /// <typeparam name="EndT"></typeparam>
    /// <returns></returns>
    public static Pattern IsSIFusionBody<T, BeginT, EndT>(string mid_name, string inputName = "input", string callName = "call", string beginName = "ld", string endName = "st")
        where T : Op
        where BeginT : Op
        where EndT : Op => IsWildcardCall<EndT>(endName, null!,
        IsWildcardCall<T>(callName, mid_name,
            IsWildcardCall<BeginT>(beginName, null!, IsWildcard(inputName))));

    /// <summary>
    /// is double input fusion body.
    /// </summary>
    /// <typeparam name="T"></typeparam>
    /// <typeparam name="T"></typeparam>
    /// <typeparam name="BeginT"></typeparam>
    /// <typeparam name="EndT"></typeparam>
    /// <returns></returns>
    public static Pattern IsDIFusionBody<T, BeginT, EndT>(string callName = "call")
        where T : Op
        where BeginT : Op
        where EndT : Op => IsWildcardCall<EndT>("st", null!,
        IsWildcardCall<T>(callName, null!,
            IsWildcardCall<BeginT>(null!, null!, IsWildcard("lhs")),
            IsWildcardCall<BeginT>(null!, null!, IsWildcard("rhs"))));

    public static Pattern IsFusion<T, BeginT, EndT>(string mid_name, string module_kind, string inputName = "input", string callName = "call", string beginName = "ld", string endName = "st", string fusionName = "fusion")
        where T : Op
        where BeginT : Op
        where EndT : Op => IsFusion(fusionName, module_kind,
        IsAlt(IsSIFusionBody<T, BeginT, EndT>(mid_name, inputName, callName, beginName, endName), IsDIFusionBody<T, BeginT, EndT>(callName)),
        WildcardVArgsPattern);

    public static Pattern IsFusion(string module_kind, Pattern body)
        => IsFusion(null, module_kind, body,
            IsVArgsRepeat(null, () => IsVar()));

    // Fusion: BeginT -> FirstOp -> SecondOp -> EndT
    public static PatternCtor IsAlt(PatternCtor patternCtorA, PatternCtor patternCtorB) => input =>
        IsAlt(patternCtorA(input), patternCtorB(input));

    public static Pattern IsPairWildcardCall<FirstOpT, SecondOpT>(string firstCallName, string secondCallName,
        Pattern input)
        where FirstOpT : Op
        where SecondOpT : Op => IsMaybeSwappableWildcardCall<SecondOpT>(
            secondCallName,
        IsMaybeSwappableWildcardCall<FirstOpT>(
            firstCallName, input));

    public static Pattern IsPairLayerFusion<FirstOpT, SecondOpT, BeginT, EndT>(
        string moduleKind,
        string firstCallName)
        where FirstOpT : Op
        where SecondOpT : Op
        where BeginT : Op
        where EndT : Op =>
        IsPairLayerFusion<FirstOpT, SecondOpT, BeginT, EndT>(moduleKind, "ld", firstCallName, "st");

    public static Pattern IsPairLayerFusion<FirstOpT, SecondOpT, BeginT, EndT>(
        string moduleKind,
        string beginCallName, string firstCallName, string endCallName)
        where FirstOpT : Op
        where SecondOpT : Op
        where BeginT : Op
        where EndT : Op => IsFusion(
            moduleKind,
        IsWildcardCall<EndT>(endCallName, null!,

            // we can't use secondCallName in getReplace because of it's optional
            IsAlt(
                input => IsPairWildcardCall<FirstOpT, SecondOpT>(firstCallName, null!, input),
                input => IsMaybeSwappableWildcardCall<FirstOpT>(firstCallName, input))(IsWildcardCall<BeginT>(beginCallName, null!, (string)null!))));

    public static Pattern IsMaybeSwappableWildcardCall<OpT>(string callName, Pattern input)
        where OpT : Op => IsMaybeSwappableWildcardCall<OpT>(callName, input, IsWildcard());

    public static Pattern IsMaybeSwappableWildcardCall<OpT>(string callName, Pattern input, Pattern swappableOther)
        where OpT : Op => IsAlt(
        IsWildcardCall<OpT>(callName, null!, input),
        IsSwappableWildcardCall<OpT>(callName, null!, input, swappableOther));
}
