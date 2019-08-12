md out
cd out
cmake .. -G "Visual Studio 16 2019" -A x64 -DNNCASE_TARGET=k210 -DCMAKE_BUILD_TYPE=Release
msbuild nncase.sln /p:Configuration=Release -m