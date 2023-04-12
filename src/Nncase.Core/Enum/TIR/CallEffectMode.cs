// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

namespace Nncase.TIR;

/// <summary>
/// The effect mode of the call.
/// </summary>
public enum CallEffectMode
{
    /// <summary>
    /// Function corresponds to an annotation(e.g. likely) and can translate to identity.
    /// </summary>
    ExprAnnotation,

    /// <summary>
    /// Pure function that do not interacts
    ///      with any external state.
    /// </summary>
    Pure,

    /// <summary>
    /// Function's that may read from states(e.g. RAM).
    /// </summary>
    ReadState,

    /// <summary>
    /// Function that may read/write from states(e.g. RAM).
    /// </summary>
    UpdateState,

    /// <summary>
    /// Opaque function, cannot make any assumption.
    /// </summary>
    Opaque,

    /// <summary>
    /// Special intrinsic to annotate call arguments info
    ///      only valid as a direct argument to a call.
    /// </summary>
    SpecialCallArg,

    /// <summary>
    /// Embed opaque information in the Expr, cannot be codegen.
    /// </summary>
    EmbedInfo,

    /// <summary>
    /// Function that changes control flow.
    /// </summary>
    ControlJump,
}
