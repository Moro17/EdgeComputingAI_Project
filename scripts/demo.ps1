<#
Start the demos on the i.MX 8M Plus HDMI screen from the laptop, through the USB debug cable (serial console):
no network and no ssh needed. On the board itself the same is just `demo final` (serial console or keyboard).

  .\scripts\demo.ps1 final      fine-tuned YuNet + tracker + look-back 3 + dark rule, 1 camera
  .\scripts\demo.ps1 cams3      the same on 3 cameras, 25 FPS each (2x2 mosaic)
  .\scripts\demo.ps1 cams4      4 cameras, detection every 2nd frame
  .\scripts\demo.ps1 baseline   per-frame detector only, no tracker / look-back: faces leak
  .\scripts\demo.ps1 stock      stock YuNet with the full pipeline
  .\scripts\demo.ps1 models     Deliverable 1 demo, cycles YuNet / fine-tuned YuNet / YOLOv8n-face every 10 s
  .\scripts\demo.ps1 stop | status | log

A second word picks the video: chokepoint, dark (ChokePoint at dark:2), mv (Music Video), nxp (default), e.g.
  .\scripts\demo.ps1 final dark        .\scripts\demo.ps1 baseline mv
  .\scripts\demo.ps1 install-videos    copy the videos built by src/make_demo_videos.py to the board (over ssh)
Starting a demo stops the one already running. Extra arguments go to edge_pipeline.py / demo.py, e.g.
  .\scripts\demo.ps1 final --video test.mp4        .\scripts\demo.ps1 final --display web   (browser, port 8080)
Options: -Port COM22 (serial console port; FTDI shows COM20-23, the console is the third one), -Ssh (use the network).
  .\scripts\demo.ps1 install    copy the code to /root/edge and link `demo` (over ssh; once, or after code changes)
#>
param([Parameter(Position = 0)][string]$Mode = "help",
      [string]$Port = "COM22",
      [switch]$Ssh,
      [Parameter(ValueFromRemainingArguments = $true)][string[]]$Rest)

$repo = Split-Path -Parent $PSScriptRoot
$cmd = ("demo $Mode " + ($Rest -join " ")).Trim()

if ($Mode -eq "install") {
    scp "$repo/src/edge_pipeline.py" "$repo/src/demo.py" "$repo/src/board_bench.py" "$repo/src/systems/anonymize.py" `
        "$repo/src/systems/bytelite.py" "$repo/scripts/demo_board.sh" imx:/root/edge/
    ssh imx "sed -i 's/\r$//' /root/edge/demo_board.sh && chmod +x /root/edge/demo_board.sh && ln -sf /root/edge/demo_board.sh /usr/bin/demo && echo installed"
    exit $LASTEXITCODE
}
if ($Mode -eq "install-videos") {  # src/make_demo_videos.py output, ~95 MB
    scp "$repo/dist/demo_videos/chokepoint.mp4" "$repo/dist/demo_videos/chokepoint_dark.mp4" "$repo/dist/demo_videos/musicvideo.mp4" "$repo/dist/demo_videos/reflection.mp4" "$repo/dist/demo_videos/chokepoint_zone.npy" "$repo/dist/demo_videos/reflection_zone.npy" imx:/root/edge/
    exit $LASTEXITCODE
}
if ($Ssh) {
    ssh imx $cmd
    exit $LASTEXITCODE
}

$sp = New-Object System.IO.Ports.SerialPort $Port, 115200, "None", 8, "One"
$sp.ReadTimeout = 500
try { $sp.Open() } catch {
    Write-Host "Cannot open $Port ($($_.Exception.Message.Trim())). Is the USB debug cable plugged in, and PuTTY closed?"
    Write-Host "Serial ports now: $([System.IO.Ports.SerialPort]::GetPortNames() -join ', ')   (pick one with -Port, or use -Ssh)"
    exit 1
}
function Read-Serial([double]$seconds, [string]$until) {
    $buf = ""; $end = (Get-Date).AddSeconds($seconds)
    while ((Get-Date) -lt $end) {
        Start-Sleep -Milliseconds 200
        $buf += $sp.ReadExisting()
        if ($until -and $buf -match $until) { break }
    }
    return $buf
}
try {
    $sp.Write("`r")
    $prompt = Read-Serial 2 'login:|Password:|[#$] *$'
    if ($prompt -match 'Password:') {
        Write-Host "The console asks for a password: log in once with PuTTY on $Port, then rerun."; exit 1
    }
    if ($prompt -match 'login:') {
        $sp.Write("root`r")
        $prompt = Read-Serial 4 'Password:|[#$] *$'
        if ($prompt -match 'Password:') { Write-Host "root needs a password here: log in with PuTTY on $Port, then rerun."; exit 1 }
    }
    $sp.Write("$cmd`r")
    $out = Read-Serial 15 '(running:|FAILED|stopped|no demo|usage:|no HDMI|^pid )[^\r\n]*[\r\n]'
    # print only the launcher's lines, not the echoed command or kernel messages
    $lines = ($out -split "`r?`n") | Where-Object { $_ -and $_ -notmatch [regex]::Escape($cmd) -and $_ -notmatch '^\s*\[\s*\d+\.\d+\]' -and $_ -notmatch '^\S*[#$] *$' }
    if ($lines) { $lines | ForEach-Object { Write-Host $_ } } else { Write-Host "(no answer from the board on $Port within 15 s)" }
    if ($out -match 'FAILED|no HDMI|usage:') { exit 1 }
} finally { $sp.Close() }
