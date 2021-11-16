## Build from source

## 从源码编译


5. Python Test

- Install dependencies
  
  ```sh
  git clone https://github.com/pythonnet/pythonnet
  cd pythonnet
  git checkout ac336a893de14aaf2c7b795568203b48030f9006
  pip install -e .
  ```

  If you need change the defualt runtime, modify the `pythonnet/__init__.py` line 18:
  ```python
  def set_default_runtime() -> None:
      set_runtime(clr_loader.get_coreclr("your-path-to/runtimeconfig.json"))
      # if sys.platform == 'win32':
      #     set_runtime(clr_loader.get_netfx())
      # else:
      #     set_runtime(clr_loader.get_mono())
  ```

  example for  `runtimeconfig.json`
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

- Setup Dotnet DLL Path

  NOTE change the env to your lib path.
  ```sh
  export NNCASE_CORE_DLL="/Users/lisa/Documents/nncase/src/Nncase.Core/bin/Debug/net6.0/Nncase.Core.dll"
  export NNCASE_IMPORTER_DLL="/Users/lisa/Documents/nncase/src/Nncase.Importer/bin/Debug/net6.0/Nncase.Importer.dll"
  export NNCASE_CLI_DLL="/Users/lisa/Documents/nncase/src/Nncase.Cli/bin/Debug/net6.0/Nncase.Cli.dll"
  export FLATBUFFERS_DLL="/Users/lisa/.nuget/packages/nncase.flatbuffers/2.0.0/lib/netstandard2.1/FlatBuffers.dll"
  export HYPERLINQ_DLL="/Users/lisa/.nuget/packages/netfabric.hyperlinq/3.0.0-beta48/lib/net6.0/NetFabric.Hyperlinq.dll"
  export HYPERLINQ_ABS_DLL="/Users/lisa/.nuget/packages/netfabric.hyperlinq.abstractions/1.3.0/lib/netstandard2.1/NetFabric.Hyperlinq.Abstractions.dll"
  ```

- Run tests

  ```sh
  cd nncase
  pytest tests
  ```