def get_source_transpose_index(perm):
    """
    transpose model output with postprocess to framework output
    """
    src_output_perm = []
    for idx, i in enumerate(perm):
        src_output_perm.append(perm.index(idx))
    return src_output_perm
