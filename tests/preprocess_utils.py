

def update_compile_options(compile_options, preprocess):
    '''
    update compile_options by preprocess options
    '''
    compile_options.preprocess = preprocess['preprocess']
    compile_options.input_layout = preprocess['input_layout']
    compile_options.output_layout = preprocess['output_layout']
    compile_options.input_type = preprocess['input_type']
    compile_options.input_shape = preprocess['input_shape']
    compile_options.input_range = preprocess['input_range']
    compile_options.swapRB = preprocess['swapRB']
    compile_options.letterbox_value = preprocess['letterbox_value']
    compile_options.mean = preprocess['mean']
    compile_options.std = preprocess['std']
    return compile_options
