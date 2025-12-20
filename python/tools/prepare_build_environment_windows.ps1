$ErrorActionPreference = "Stop"

pip install --no-cache-dir `
    https://repo.radeon.com/rocm/windows/rocm-rel-7.1.1/rocm_sdk_core-0.1.dev0-py3-none-win_amd64.whl `
    https://repo.radeon.com/rocm/windows/rocm-rel-7.1.1/rocm_sdk_devel-0.1.dev0-py3-none-win_amd64.whl `
    https://repo.radeon.com/rocm/windows/rocm-rel-7.1.1/rocm_sdk_libraries_custom-0.1.dev0-py3-none-win_amd64.whl `
    https://repo.radeon.com/rocm/windows/rocm-rel-7.1.1/rocm-0.1.dev0.tar.gz
rocm-sdk init
$env:ROCM_PATH = $(python -c "from rocm_sdk._devel import get_devel_root;print(get_devel_root())")

$env:PATH = "$env:ROCM_PATH\bin;$env:PATH"
$env:CC = "$env:ROCM_PATH\lib\llvm\bin\clang.exe"
$env:CXX = "$env:ROCM_PATH\lib\llvm\bin\clang++.exe"

$env:HIP_PLATFORM = "amd"
$env:HIP_PATH = "$env:ROCM_PATH"
$env:HIP_DEVICE_LIB_PATH = "$env:ROCM_PATH\lib\llvm\amdgcn\bitcode"
$env:HIP_CLANG_ROOT = "$env:ROCM_PATH\lib\llvm\"
if ($env:ROCM_ARCH -eq "all") {
    $env:PYTORCH_ROCM_ARCH = "gfx1100;gfx1101;gfx1102;gfx1150;gfx1151;gfx1200;gfx1201"
} else {
    $env:PYTORCH_ROCM_ARCH = "$env:ROCM_ARCH"
}

(New-Object Net.WebClient).DownloadFile("https://aka.ms/vs/17/release/vs_buildtools.exe", "$(Get-Location)\vs_buildtools.exe")
Start-Process ".\vs_buildtools.exe" -ArgumentList "--passive", "--wait", "--addProductLang", "En-us", "--channelId", "VisualStudio.17.Release", "--add", "Microsoft.VisualStudio.Workload.VCTools;includeRecommended", "--add", "Microsoft.VisualStudio.Component.VC.ATL" -NoNewWindow -Wait | Out-Default
Remove-Item ".\vs_buildtools.exe"
& "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\Tools\Launch-VsDevShell.ps1" -Arch amd64

(New-Object Net.WebClient).DownloadFile("https://registrationcenter-download.intel.com/akdlm/IRC_NAS/1f18901e-877d-469d-a41a-a10f11b39336/intel-oneapi-base-toolkit-2025.3.0.372_offline.exe", "$(Get-Location)\oneapi-installer.exe")
Start-Process ".\oneapi-installer.exe" -ArgumentList "-s", "-x", "-f", ".\oneapi_extracted", "--log", "extract.log" -NoNewWindow -Wait | Out-Default
Remove-Item ".\oneapi-installer.exe"
Start-Process ".\oneapi_extracted\bootstrapper.exe" -ArgumentList "-s", "--action", "install", "--components=intel.oneapi.win.mkl.devel", "--eula", "accept", "-p=NEED_VS2017_INTEGRATION=0", "-p=NEED_VS2019_INTEGRATION=0", "-p=NEED_VS2022_INTEGRATION=0", "-p=NEED_VS2026_INTEGRATION=0", "--log-dir=." -NoNewWindow -Wait | Out-Default
Remove-Item .\oneapi_extracted -Recurse

$ONEDNN_VERSION="3.10.2"
(New-Object Net.WebClient).DownloadFile("https://github.com/oneapi-src/oneDNN/archive/refs/tags/v$ONEDNN_VERSION.zip", "$(Get-Location)\oneDNN.zip")
Expand-Archive .\oneDNN.zip .
Remove-Item .\oneDNN.zip
Set-Location ".\oneDNN-$ONEDNN_VERSION"
cmake -DCMAKE_BUILD_TYPE=Release -DONEDNN_LIBRARY_TYPE=STATIC -DONEDNN_BUILD_EXAMPLES=OFF -DONEDNN_BUILD_TESTS=OFF -DONEDNN_ENABLE_WORKLOAD=INFERENCE -DONEDNN_ENABLE_PRIMITIVE="CONVOLUTION;REORDER" -DONEDNN_BUILD_GRAPH=OFF .
cmake --build . --config Release --target install --parallel 6
Set-Location ..\
Remove-Item ".\oneDNN-$ONEDNN_VERSION" -Recurse

$options = ""
if (-not $env:PYTORCH_ROCM_ARCH.Contains(";")) {
    $targets = $env:PYTORCH_ROCM_ARCH.Substring(0, $env:PYTORCH_ROCM_ARCH.Length-2)
    Start-Process python -ArgumentList ".\third_party\composable_kernel\example\ck_tile\01_fmha\generate.py", "-d", "fwd", "-f", "*batch*nlogits*nbias*nmask*nlse*ndropout*nsink*", "--output_dir", "./src/ops/flash-attn-ck", "--receipt", "2", "--optdim", "32,64,128,256", "--targets", $targets -NoNewWindow -Wait | Out-Default
    $options += " -DWITH_FLASH_ATTN=ON"
}
cmake -GNinja -DCMAKE_BUILD_TYPE=Release -S . -B build -DCMAKE_CXX_FLAGS="-Wno-deprecated-literal-operator" -DCMAKE_INSTALL_PREFIX="$env:CTRANSLATE2_ROOT" -DCMAKE_PREFIX_PATH="C:\Program Files (x86)\Intel\oneAPI\compiler\latest\lib;C:\Program Files (x86)\oneDNN" -DBUILD_CLI=OFF -DWITH_DNNL=ON -DWITH_HIP=ON -DCMAKE_HIP_ARCHITECTURES="$env:PYTORCH_ROCM_ARCH" -DBUILD_TESTS=ON $options.Trim()

cmake --build build --config Release --target install --parallel 6 --verbose

Copy-Item "$env:CTRANSLATE2_ROOT\bin\ctranslate2.dll" ".\python\ctranslate2"
Copy-Item ".\README.md" ".\python"
Copy-Item "C:\Program Files (x86)\Intel\oneAPI\compiler\latest\bin\libiomp5md.dll" ".\python\ctranslate2"
Copy-Item "$env:ROCM_PATH\bin\amdhip64_7.dll" ".\python\ctranslate2"
Copy-Item "$env:ROCM_PATH\bin\amd_comgr0701.dll" ".\python\ctranslate2"

Compress-Archive -Path ".\build\bin\*.dll", ".\python\ctranslate2\*.dll", ".\build\tests\*.exe" -DestinationPath tests.zip
