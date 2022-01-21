# Build

```sh
cd csharp
dotnet build -p:TargetArchitecture=arm64 # your arch
```

# Pack

```sh
# rm -rf ~/.nuget/packages/nncase.runtime.native*
nuget pack nuget/Nncase.Runtime.Native.osx-arm64.nuspec -OutputDirectory ../nupkg
nuget pack nuget/Nncase.Runtime.Native.nuspec -OutputDirectory ../nupkg
```