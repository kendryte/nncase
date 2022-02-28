# Build

```sh
cd csharp
dotnet build -c Release -p:TargetArchitecture=arm64 # your arch
```

# Pack Single

```sh
# rm -rf ~/.nuget/packages/nncase.simulator.native*
# nuget pack nuget/Nncase.Simulator.Native.osx-arm64.nuspec -OutputDirectory ../nupkg
nuget pack nuget/Nncase.Simulator.Native.linux-x64.nuspec -OutputDirectory ../nupkg
nuget pack nuget/Nncase.Simulator.Native.win-x64.nuspec -OutputDirectory ../nupkg
nuget pack nuget/Nncase.Simulator.Native.osx-x64.nuspec -OutputDirectory ../nupkg
```

# Pack All

```sh
nuget pack nuget/Nncase.Simulator.Native.nuspec -OutputDirectory ../nupkg
```

# Upload

```sh
nuget push Nncase.Simulator.Native.1.0.0.nupkg <apikey> -Source https://www.myget.org/F/zhen8838/api/v3/index.json
nuget push Nncase.Simulator.Native.linux-x64.1.0.0.nupkg <apikey> -Source https://www.myget.org/F/zhen8838/api/v3/index.json
nuget push Nncase.Simulator.Native.osx-arm64.1.0.0.nupkg <apikey> -Source https://www.myget.org/F/zhen8838/api/v3/index.json
nuget push Nncase.Simulator.Native.osx-x64.1.0.0.nupkg <apikey> -Source https://www.myget.org/F/zhen8838/api/v3/index.json
nuget push Nncase.Simulator.Native.win-x64.1.0.0.nupkg <apikey> -Source https://www.myget.org/F/zhen8838/api/v3/index.json
```