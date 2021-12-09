## Build from source

## 从源码编译


5. Python Test

- Install dependencies
  PythonNet对于选择coreclr的功能目前只有dev版本才可使用,因此需要手动下载安装.
  ```sh
  git clone https://github.com/pythonnet/pythonnet
  cd pythonnet
  git checkout ac336a893de14aaf2c7b795568203b48030f9006
  pip install -e .
  ```

  当安装pythonnet后,可以直接修改`pythonnet/__init__.py` line 18选择需要的runtime:

  ```python
  def set_default_runtime() -> None:
      set_runtime(clr_loader.get_coreclr("your-path-to/runtimeconfig.json"))
      # if sys.platform == 'win32':
      #     set_runtime(clr_loader.get_netfx())
      # else:
      #     set_runtime(clr_loader.get_mono())
  ```
  注释掉原有的set_runtime并且添加上自己的设置代码
  其中runtimeconfig.json需要绝对路径

  the example of `runtimeconfig.json`
  ```json
  {
    "runtimeOptions": {
      "tfm": "net6.0",
      "framework": {
        "name": "Microsoft.NETCore.App",
        "version": "6.0.0"
      }
    }
  }
  ```
  framework中的版本号是dotnet runtime的版本号，可以通过dotnet --info查看

- 设置dotnet dll Path
  由于不同pc上安装的dotnet package不同,因此开发Nncase时需要手动提供DLL PATH.
  ```sh
  "PYTHONPATH": "${workspaceFolder}/python:${workspaceFolder}/tests:${env:PYTHONPATH}",
  "PYTHONNET_PYDLL": "/Users/lisa/mambaforge/lib/libpython3.9.dylib",
  "NNCASE_CLI": "${workspaceFolder}/bin/Nncase.Cli/net6.0",
  ```

- Run tests

  ```sh
  cd nncase
  pytest tests
  ```