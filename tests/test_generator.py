import sys
from abc import ABCMeta, abstractmethod


def join_by_breakline(data):
    return "\n".join(data)


class TestGenerator(metaclass=ABCMeta):
    def __init__(self):
        super().__init__()
        self.import_list = []
        self.op_libary = ''

    def add_import_list(self, import_list=[]):
        self.import_list = self.import_list + import_list

    def set_test_runner(self, runner, fun):
        self.test_runner = runner
        self.add_import_list([f"from test_runner import {runner}"])
        self.runner_fun = fun

    def generate_params_str(self, params):
        return ", ".join(params)

    def generate_license(self):
        return f"""# Copyright 2019-2021 Canaan Inc.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""

    def generate_annotations(self, name):
        return f"""\"\"\"System test: test {name}\"\"\"
# pylint: disable=invalid-name, unused-argument, import-outside-toplevel
"""

    def generate_import(self):
        def generate_other_import(import_list):
            return "\n".join(import_list)

        # TODO:generate optional
        def _generate_import(self):
            if len(self.import_list) != 0:
                return generate_other_import(self.import_list)
            else:
                return self.generate_raise_need_changed('has import all package?')

        return f"""import pytest
{_generate_import(self)}
"""

    def generate_raise_need_changed(self, check_info):
        return f"""raise Exception("need check and changed:{check_info}")"""

    def generate_make_module(self, name, params):
        title_name = name.title()
        module_name = f"{title_name}Module"
        params_str = self.generate_params_str(params)
        return f"""def _make_module({params_str}):
    class {module_name}(tf.Module):
        def __init__(self):
            super({title_name}Module).__init__()
            {self.generate_raise_need_changed("may be need some setting")}

        @tf.function(input_signature=[tf.TensorSpec(in_shape, tf.float32)])
        def __call__(self, x):
            {self.generate_raise_need_changed("1:tf function;"
                                              "2:op name and test name may be not same;"
                                              "3:call param may be need to specify name")}
            return tf.{name}(x, {params_str})
    return {module_name}()
"""

    def generate_input_frame(self, params):
        def generate_one_input_frame(param):
            return f"""{param} = [

]
    """

        return join_by_breakline(map(generate_one_input_frame, params))

    def generate_mark_parametrizes(self, params):
        def generate_mark_parametrize(param):
            return f"""@pytest.mark.parametrize('{param}', {param})"""

        return join_by_breakline(map(generate_mark_parametrize, params))

    def generate_run_test(self, name, params):
        params_str = ",".join(params)
        return f"""
def test_{name}({params_str}, request):
    module = _make_module({params_str})
    runner = TfliteTestRunner(request.node.name)
    model_file = runner.from_tensorflow(module)
    runner.run(model_file)
"""

    def generate_test_fun(self, name, params):
        return self.generate_mark_parametrizes(params) + self.generate_run_test(name, params)

    def generate_main(self, name):
        return f"""if __name__ == "__main__":
    pytest.main(['-vv', 'test_{name}.py'])
"""

    def generate_test_src(self, name, params):
        return join_by_breakline([self.generate_license() + self.generate_annotations(name),
                                  self.generate_import(),
                                  self.generate_make_module(name, params), self.generate_input_frame(params),
                                  self.generate_test_fun(name, params), self.generate_main(name)])

    def generate_test_file(self, test_name, params):
        params[0] = 'in_shape'
        test_file = open(f"test_{test_name}.py", 'w+')
        src = self.generate_test_src(test_name, params)
        test_file.write(src)


class TensorflowTestGenerator(TestGenerator):
    def __init__(self):
        super().__init__()
        self.op_libary = 'tensorflow'
        self.set_test_runner('TfliteTestRunner', 'from_tensorflow')
        self.add_import_list(
            ['import tensorflow as tf'])


# TODO:now only support tensorflow
# TODO:name assign? params=params, indices=indices
# odd or even
# forward or call
# if set value, then not raise except
if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("please input test name and params")
        print("python generate_test.py test_name param1 param2 ...")
        exit(-1)
    TensorflowTestGenerator().generate_test_file(sys.argv[1], sys.argv[2:])
