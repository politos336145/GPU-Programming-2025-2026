# build_win.ps1 - Windows native build script for Angry Santa (Vulkan + CUDA)
# Usage: .\build_win.ps1

$ErrorActionPreference = "Stop"

# ---- Paths ----
$scriptDir  = Split-Path -Parent $MyInvocation.MyCommand.Path
$projectDir = Split-Path -Parent $scriptDir
$buildDir   = Join-Path $projectDir "build"
$_machinePath = [System.Environment]::GetEnvironmentVariable("PATH", "Machine")
$_userPath    = [System.Environment]::GetEnvironmentVariable("PATH", "User")

# ---- MODIFY HERE: Paths to tools & SDK ----
$cmakePath = "C:\Program Files\Microsoft Visual Studio\2022\Professional\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe"
$ninjaPath = "C:\Program Files\Microsoft Visual Studio\2022\Professional\Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja\ninja.exe"
$msvcRoot = "C:\Program Files\Microsoft Visual Studio\2022\Professional"
$msvc = "$msvcRoot\VC\Tools\MSVC\14.44.35207"
$msvcBin = "$msvc\bin\HostX64\x64"
$msvcInclude = "$msvc\include"
$msvcLib = "$msvc\lib\x64"
$windowsSdkRoot = "C:\Program Files (x86)\Windows Kits\10"
$windowsSdkBin = "$windowsSdkRoot\bin\10.0.26100.0\x64"
$windowsSdkLib = "$windowsSdkRoot\Lib\10.0.26100.0\ucrt\x64;$windowsSdkRoot\Lib\10.0.26100.0\um\x64"
$windowsSdkInclude = "$windowsSdkRoot\Include\10.0.26100.0\ucrt;$windowsSdkRoot\Include\10.0.26100.0\um;$windowsSdkRoot\Include\10.0.26100.0\shared"
$cudaPath  = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8"
$vulkanSdk = "C:\VulkanSDK\1.4.341.1"

$env:PATH = "$cudaPath\bin;$windowsSdkBin;$msvcBin;$_machinePath;$_userPath" # Avoid the Windows cmd "input line too long" / vcvars64.bat failure (~8191 char limit)
$env:LIB = "$msvcLib;$windowsSdkLib"
$env:INCLUDE = "$msvcInclude;$windowsSdkInclude"
$env:VSINSTALLDIR = "$msvcRoot\"
$env:VCToolsInstallDir = "$msvc"
$env:CUDA_PATH  = "$cudaPath"
$env:VULKAN_SDK = "$vulkanSdk"

# ---- Clean & Configure ----
Write-Host "=== Configuring CMake ===" -ForegroundColor Cyan
if (Test-Path $buildDir) { Remove-Item -Recurse -Force $buildDir }
New-Item -ItemType Directory $buildDir | Out-Null
Set-Location $buildDir

& $cmakePath .. `
    -DENABLE_VULKAN=ON `
    -DENABLE_NVTX=ON `
    -DENABLE_CUDA_LINEINFO=ON `
    -DENABLE_CUDA_PTXAS_VERBOSE=ON `
    -G Ninja `
    -DCMAKE_BUILD_TYPE=Release `
    -DCMAKE_MAKE_PROGRAM="$ninjaPath" `
    "-DCMAKE_CUDA_FLAGS=-allow-unsupported-compiler -Xcompiler=/D_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH" `
    "-DCMAKE_CXX_FLAGS=/D_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH"
if ($LASTEXITCODE -ne 0) { Write-Host "CMake configure FAILED" -ForegroundColor Red; exit 1 }

# ---- Build ----
Write-Host "`n=== Building ===" -ForegroundColor Cyan
& $cmakePath --build . --parallel
if ($LASTEXITCODE -ne 0) { Write-Host "Build FAILED" -ForegroundColor Red; exit 1 }

Write-Host "`n=== BUILD SUCCESSFUL ===" -ForegroundColor Green
Write-Host "Executable: $buildDir\AngrySanta_GPU.exe"
Write-Host "Run headless:    .\build\AngrySanta_GPU.exe"
Write-Host "Run with Vulkan: .\build\AngrySanta_GPU.exe --render"
Write-Host ""

Set-Location $projectDir
