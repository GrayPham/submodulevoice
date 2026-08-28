@echo off
rem ---------------------------------------------------------------
rem  omnivoice.cpp - Windows build helper
rem
rem    build-win.cmd cpu     -> CPU only  (build-cpu\)   <- weak machines
rem    build-win.cmd cuda    -> CPU+CUDA  (build-cuda\)  <- NVIDIA
rem
rem  GGML_BACKEND_DL=ON       : backends are .dll loaded at runtime, so the
rem                             same binary degrades to CPU when no GPU.
rem  GGML_CPU_ALL_VARIANTS=ON : ships SSE4.2 / AVX / AVX2 / AVX512 kernels
rem                             and picks one at runtime -> old CPUs work.
rem  OMNIVOICE_SHARED=ON      : also emits omnivoice.dll for the Python
rem                             ctypes binding (ov_* C ABI).
rem ---------------------------------------------------------------
setlocal

set BACKEND=%1
if "%BACKEND%"=="" set BACKEND=cpu

set VS_BUILDTOOLS=C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat
set VCVARS=C:\Program Files\Microsoft Visual Studio\2022\Professional\VC\Auxiliary\Build\vcvars64.bat
if not exist "%VCVARS%" set VCVARS=%VS_BUILDTOOLS%
if /I "%1"=="cuda" goto skipvcvars
call "%VCVARS%"
:skipvcvars

set NINJA=C:\Program Files\Microsoft Visual Studio\2022\Professional\Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja\ninja.exe
if not exist "%NINJA%" set NINJA=C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja\ninja.exe

cd /d "%~dp0omnivoice.cpp"

rem omnivoice.cpp la repo rieng, .gitignore cua ho chi co build/ trong khi ta
rem dat ten build-cpu/ va build-cuda/. Khong khai bao thi git cua ho bao ban
rem hang tram file. Ghi vao info/exclude: ignore rieng cua may minh, khong
rem dung vao .gitignore cua upstream, khong mat khi pull.
if exist ".git\info\exclude" (
    findstr /C:"build-*/" ".git\info\exclude" >nul 2>&1 || (
        echo.>> ".git\info\exclude"
        echo # thu muc build cua submodulevoice/build-win.cmd>> ".git\info\exclude"
        echo build-*/>> ".git\info\exclude"
        echo [build-win] da them build-*/ vao omnivoice.cpp\.git\info\exclude
    )
)

if /I "%BACKEND%"=="cuda" goto cuda

:cpu
cmake -B build-cpu -G Ninja -DCMAKE_MAKE_PROGRAM="%NINJA%" -DCMAKE_BUILD_TYPE=Release -DGGML_CPU_ALL_VARIANTS=ON -DGGML_BACKEND_DL=ON -DOMNIVOICE_SHARED=ON
if errorlevel 1 exit /b 1
cmake --build build-cpu -j %NUMBER_OF_PROCESSORS%
if errorlevel 1 exit /b 1
echo === done -^> omnivoice.cpp\build-cpu ===
goto :eof

:cuda
rem CUDA 12.1 + MSVC 14.41 is a dead end: nvcc's host_config.h rejects
rem _MSC_VER >= 1940, and even with -allow-unsupported-compiler the 14.41 STL
rem hard-static_asserts "expected CUDA 12.4 or newer" (yvals_core.h).
rem So the CUDA build pins the v142 toolset (MSVC 19.29), which nvcc 12.1
rem accepts natively. Install CUDA >= 12.8 to drop this pin.
call "%VS_BUILDTOOLS%" -vcvars_ver=14.29
if errorlevel 1 exit /b 1
cd /d "%~dp0omnivoice.cpp"

rem Danh sách kiến trúc GPU. Mặc định phủ từ Pascal (GTX 10xx) đến Ada, cộng
rem PTX của 61 để JIT được trên card lạ. Build lâu hơn nhiều so với một kiến
rem trúc, nhưng exe phát hành phải chạy được trên máy người khác — sm_89 đơn lẻ
rem chỉ chạy trên RTX 40xx.
rem   61 = GTX 1050/1060/1080   75 = GTX 1650, RTX 20xx
rem   86 = RTX 30xx             89 = RTX 40xx
if "%CUDA_ARCHS%"=="" set CUDA_ARCHS=61-virtual;61-real;75-real;86-real;89-real
cmake -B build-cuda -G Ninja -DCMAKE_MAKE_PROGRAM="%NINJA%" -DCMAKE_BUILD_TYPE=Release -DGGML_CPU_ALL_VARIANTS=ON -DGGML_BACKEND_DL=ON -DOMNIVOICE_SHARED=ON -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES="%CUDA_ARCHS%"
if errorlevel 1 exit /b 1
cmake --build build-cuda -j %NUMBER_OF_PROCESSORS%
if errorlevel 1 exit /b 1
echo === done -^> omnivoice.cpp\build-cuda ===
goto :eof
