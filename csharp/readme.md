# Build

```sh
cd csharp
dotnet build -c Release -p:TargetArchitecture=arm64 # your arch
```

# Pack

```sh
# rm -rf ~/.nuget/packages/nncase.simulator.native*
# nuget pack nuget/Nncase.Simulator.Native.osx-arm64.nuspec -OutputDirectory ../nupkg
nuget pack nuget/Nncase.Simulator.Native.linux-x64.nuspec -OutputDirectory ../nupkg
nuget pack nuget/Nncase.Simulator.Native.nuspec -OutputDirectory ../nupkg
```