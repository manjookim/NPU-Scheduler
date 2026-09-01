# Cityscapes 다운로드 (PowerShell 전용, bash/WSL 불필요)
# 계정 필요: https://www.cityscapes-dataset.com/register/
#
# 사용:
#   .\download_cityscapes.ps1 -Dest D:\datasets\Cityscapes
#
# 자격증명은 실행 중에 물어보므로 명령 기록(PSReadLine history)에 남지 않는다.
# Windows 10 1803+ 에 기본 포함된 curl.exe 를 쓴다 (-C - 로 이어받기 지원).

param(
    [string]$Dest = "D:\datasets\Cityscapes"
)

$ErrorActionPreference = "Stop"

# curl.exe 존재 확인 (PowerShell 별칭 curl 이 아니라 진짜 실행파일)
$curl = Join-Path $env:SystemRoot "System32\curl.exe"
if (-not (Test-Path $curl)) { throw "curl.exe 를 찾을 수 없습니다. Windows 10 1803 이상이 필요합니다." }

New-Item -ItemType Directory -Force -Path $Dest | Out-Null
Set-Location $Dest
Write-Host "저장 위치: $Dest" -ForegroundColor Cyan

$user = Read-Host "Cityscapes 아이디(이메일)"
$sec  = Read-Host "Cityscapes 비밀번호" -AsSecureString
$pass = [Runtime.InteropServices.Marshal]::PtrToStringAuto(
        [Runtime.InteropServices.Marshal]::SecureStringToBSTR($sec))

# 비밀번호에 & = + 같은 문자가 있으면 POST 본문이 깨지므로 URL 인코딩한다
$userEnc = [uri]::EscapeDataString($user)
$passEnc = [uri]::EscapeDataString($pass)
$pass = $null

Write-Host "`n[0/3] 로그인" -ForegroundColor Yellow
& $curl -s -c cookies.txt -d "username=$userEnc&password=$passEnc&submit=Login" `
        "https://www.cityscapes-dataset.com/login/" -o $env:TEMP\cs_login.html
$passEnc = $null

# 로그인 실패하면 이후 다운로드가 HTML 파일을 내려받아 조용히 망가지므로 여기서 검사
if (-not (Select-String -Path cookies.txt -Pattern "PHPSESSID" -Quiet)) {
    throw "로그인 실패 — 아이디/비밀번호를 확인하세요. (세션 쿠키 없음)"
}
Write-Host "  로그인 OK" -ForegroundColor Green

Write-Host "`n[1/3] gtFine_trainvaltest.zip (약 253MB)" -ForegroundColor Yellow
& $curl -L -C - -b cookies.txt -o gtFine_trainvaltest.zip `
        "https://www.cityscapes-dataset.com/file-handling/?packageID=1"

Write-Host "`n[2/3] leftImg8bit_trainvaltest.zip (약 11GB) — 오래 걸립니다" -ForegroundColor Yellow
& $curl -L -C - -b cookies.txt -o leftImg8bit_trainvaltest.zip `
        "https://www.cityscapes-dataset.com/file-handling/?packageID=3"

# 받은 게 진짜 zip 인지 확인 (로그인 만료 시 HTML 이 내려올 수 있음)
foreach ($z in @("gtFine_trainvaltest.zip", "leftImg8bit_trainvaltest.zip")) {
    $sig = [System.IO.File]::ReadAllBytes((Resolve-Path $z))[0..1]
    if ($sig[0] -ne 0x50 -or $sig[1] -ne 0x4B) {
        throw "$z 가 zip 이 아닙니다 (로그인 만료 추정). 파일을 지우고 다시 실행하세요."
    }
}

Write-Host "`n[3/3] 압축 해제" -ForegroundColor Yellow
tar -xf gtFine_trainvaltest.zip
tar -xf leftImg8bit_trainvaltest.zip
Remove-Item cookies.txt -Force

$tr = (Get-ChildItem -Recurse -Filter *_leftImg8bit.png leftImg8bit\train).Count
$va = (Get-ChildItem -Recurse -Filter *_leftImg8bit.png leftImg8bit\val).Count
Write-Host "`n완료" -ForegroundColor Green
Write-Host ("  train {0}장 (정상=2975)  {1}" -f $tr, $(if ($tr -eq 2975) {"OK"} else {"확인 필요"}))
Write-Host ("  val   {0}장 (정상=500)   {1}" -f $va, $(if ($va -eq 500)  {"OK"} else {"확인 필요"}))
Write-Host "`n다음: python train_deeplab_cityscapes.py --data `"$Dest`" --out .\runs\dlv3_mnv2 --epochs 40 --batch 4"
