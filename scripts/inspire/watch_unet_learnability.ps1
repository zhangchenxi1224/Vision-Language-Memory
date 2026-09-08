param(
    [Parameter(Mandatory=$true)][string]$Commit,
    [Parameter(Mandatory=$true)][string]$RemoteRepository,
    [Parameter(Mandatory=$true)][string]$RemoteOutput,
    [Parameter(Mandatory=$true)][string]$LogDirectory,
    [int]$MaxWaitMinutes=120
)
$ErrorActionPreference='Stop'
New-Item -ItemType Directory -Path $LogDirectory -Force | Out-Null
$instanceName='vlm-unet-fit-h200x2-20260909'
$inspirePrefix=@('-d','Ubuntu','--cd','/tmp','--exec','env',
 'INSPIRE_REQUESTS_HTTP_PROXY=http://127.0.0.1:7897',
 'INSPIRE_REQUESTS_HTTPS_PROXY=http://127.0.0.1:7897',
 '/home/zhangchenxi/.local/bin/inspire','--no-env-file')
$deadlineUtc=[DateTime]::UtcNow.AddMinutes($MaxWaitMinutes)
$ptyHelper=(Join-Path $PSScriptRoot 'inspire_pty_exec.py')
$ptyHelperLinux=(& wsl.exe -d Ubuntu --exec wslpath -a $ptyHelper | Out-String).Trim()
$pythonTemplate="import subprocess,sys; sys.exit(subprocess.call(['python3','$RemoteRepository/scripts/inspire/launch_unet_learnability.py','--commit','$Commit','--output','$RemoteOutput','--wait-for-warmup','--lease-minutes','__LEASE__']))"
while([DateTime]::UtcNow -lt $deadlineUtc){
    $statusText=(& wsl.exe @inspirePrefix notebook status $instanceName --workspace 分布式训练空间 2>&1 | Out-String)
    $statusCode=$LASTEXITCODE
    $statusText | Set-Content -LiteralPath (Join-Path $LogDirectory 'platform-status.txt') -Encoding utf8
    if($statusCode -eq 0 -and $statusText -match '(?m)^Status:\s+RUNNING\s*$'){
        $leaseMinutes=165
        if($statusText -match '(?m)^Auto-stop In:\s+(.+)$'){
            $timeText=$Matches[1]
            $remainingMinutes=0
            if($timeText -match '(\d+)h'){$remainingMinutes+=60*[int]$Matches[1]}
            if($timeText -match '(\d+)m'){$remainingMinutes+=[int]$Matches[1]}
            $leaseMinutes=[Math]::Min(165,$remainingMinutes-10)
        }
        if($leaseMinutes -lt 5){
            @{state='insufficient_allocation_lifetime';instance=$instanceName} | ConvertTo-Json |
                Set-Content -LiteralPath (Join-Path $LogDirectory 'status.json') -Encoding utf8
            exit 4
        }
        $pythonCode=$pythonTemplate.Replace('__LEASE__',([string]$leaseMinutes))
        $hex=[Convert]::ToHexString([Text.Encoding]::UTF8.GetBytes($pythonCode)).ToLowerInvariant()
        $remoteCommand="python3 -c 'exec(bytes.fromhex(`"$hex`"))'"
        $launchText=(& wsl.exe -d Ubuntu --cd /tmp --exec python3 $ptyHelperLinux notebook exec $instanceName --workspace 分布式训练空间 --timeout 90 $remoteCommand 2>&1 | Out-String)
        $launchCode=$LASTEXITCODE
        $launchText | Set-Content -LiteralPath (Join-Path $LogDirectory 'launch.txt') -Encoding utf8
        if($launchText -match 'waiting_for_warmup_boundary'){
            @{state='allocation_ready_waiting_for_warmup_boundary';instance=$instanceName;epoch=[DateTimeOffset]::UtcNow.ToUnixTimeSeconds()} | ConvertTo-Json |
                Set-Content -LiteralPath (Join-Path $LogDirectory 'status.json') -Encoding utf8
            Start-Sleep -Seconds 30
            continue
        }
        @{state= $(if($launchCode -eq 0){'launch_dispatched'}else{'launch_failed'});epoch=[DateTimeOffset]::UtcNow.ToUnixTimeSeconds();exit_code=$launchCode;instance=$instanceName;output=$RemoteOutput;commit=$Commit} |
            ConvertTo-Json | Set-Content -LiteralPath (Join-Path $LogDirectory 'status.json') -Encoding utf8
        exit $launchCode
    }
    if($statusCode -eq 0 -and $statusText -match '(?m)^Status:\s+(FAILED|STOPPED|DELETED)\s*$'){
        @{state='allocation_unavailable';detail=$statusText;instance=$instanceName} | ConvertTo-Json |
            Set-Content -LiteralPath (Join-Path $LogDirectory 'status.json') -Encoding utf8
        exit 2
    }
    @{state='waiting_for_allocation';instance=$instanceName;epoch=[DateTimeOffset]::UtcNow.ToUnixTimeSeconds();deadline=$deadlineUtc.ToString('o');output=$RemoteOutput;commit=$Commit} |
        ConvertTo-Json | Set-Content -LiteralPath (Join-Path $LogDirectory 'status.json') -Encoding utf8
    Start-Sleep -Seconds 30
}
@{state='queue_watch_timeout';instance=$instanceName;output=$RemoteOutput} | ConvertTo-Json |
    Set-Content -LiteralPath (Join-Path $LogDirectory 'status.json') -Encoding utf8
exit 3
