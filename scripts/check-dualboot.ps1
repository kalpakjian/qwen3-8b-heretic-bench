#Requires -Version 5.1
<#
.SYNOPSIS
    Read-only pre-flight check before dual-booting Omarchy alongside Windows 11.

.DESCRIPTION
    Touches nothing. Reports:
      - BitLocker / Device Encryption state  (CRITICAL - key backup first)
      - Secure Boot + TPM state
      - Fast Startup
      - Disk layout and where each EFI System Partition lives
      - Whether an existing Linux (Omarchy/Arch) install is already visible

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File .\check-dualboot.ps1
#>

[CmdletBinding()]
param()

$ErrorActionPreference = 'Continue'

function Section([string]$Title) {
    Write-Host ""
    Write-Host ("=" * 68) -ForegroundColor DarkGray
    Write-Host "  $Title" -ForegroundColor Cyan
    Write-Host ("=" * 68) -ForegroundColor DarkGray
}
function Ok([string]$m)   { Write-Host "  [ OK ]  $m" -ForegroundColor Green }
function Warn([string]$m) { Write-Host "  [WARN]  $m" -ForegroundColor Yellow }
function Bad([string]$m)  { Write-Host "  [FAIL]  $m" -ForegroundColor Red }
function Info([string]$m) { Write-Host "          $m" -ForegroundColor Gray }

$isAdmin = ([Security.Principal.WindowsPrincipal] `
    [Security.Principal.WindowsIdentity]::GetCurrent()
).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)

Write-Host ""
Write-Host "  Dual-boot pre-flight check" -ForegroundColor White
Write-Host "  Read-only. Nothing on this system will be modified." -ForegroundColor DarkGray
if (-not $isAdmin) {
    Warn "Not elevated - BitLocker and firmware checks may be incomplete."

# ---------------------------------------------------------------- BitLocker --
Section "BitLocker / Device Encryption  (most important)"

$blChecked = $false
$blFindings = @()
try {
    $vols = Get-BitLockerVolume -ErrorAction Stop
    $blChecked = $true
    foreach ($v in $vols) {
        $line = "Volume $($v.MountPoint) : Protection=$($v.ProtectionStatus)  Status=$($v.VolumeStatus)  Encrypted=$($v.EncryptionPercentage)%"
        if ($v.ProtectionStatus -eq 'On') {
            $blFindings += $v.MountPoint
            Warn $line
            foreach ($p in $v.KeyProtector) { Info "key protector: $($p.KeyProtectorType)" }
        } else {
            Ok $line
        }
    }
} catch {
    Warn "Get-BitLockerVolume unavailable: $($_.Exception.Message)"
    if (-not $isAdmin) {
        Info "This is almost certainly the missing elevation, not a missing feature."
    }
}

if (-not $blChecked) {
    $m = & manage-bde -status 2>&1 | Out-String
    if ($LASTEXITCODE -eq 0 -and $m -match 'Conversion Status') {
        $blChecked = $true
        if ($m -match 'Protection On') { $blFindings += 'manage-bde reports Protection On' }
        Write-Host $m
    }
}

if (-not $blChecked) {
    Write-Host ""
    Bad "UNKNOWN - could not determine BitLocker state."
    Info "Do NOT assume it is off. Re-run this script as Administrator:"
    Info "    Start-Process powershell -Verb RunAs -ArgumentList '-ExecutionPolicy Bypass -File .\check-dualboot.ps1'"
    Info "Or check manually:  manage-bde -status"
} elseif ($blFindings.Count -gt 0) {
    Write-Host ""
    Bad "BitLocker is ON for: $($blFindings -join ', ')"
    Info "Changing Secure Boot or TPM state can trigger a recovery prompt."
    Info "BACK UP THE 48-DIGIT KEY BEFORE TOUCHING BIOS:"
    Info "    manage-bde -protectors -get C:"
    Info "    https://account.microsoft.com/devices/recoverykey"
} else {
    Write-Host ""
    Ok "BitLocker is confirmed OFF / unprovisioned."
}

# ------------------------------------------------------------ Secure Boot ----
Section "Secure Boot"

try {
    $sb = Confirm-SecureBootUEFI -ErrorAction Stop
    if ($sb) {
        Warn "Secure Boot is ENABLED."
        Info "Omarchy's installer requires Secure Boot and/or TPM to be disabled."
        Info "Expect a Windows 11 validation prompt after disabling."
    } else {
        Ok "Secure Boot is already disabled."
    }
} catch {
    Info "Could not query Secure Boot: $($_.Exception.Message)"
    Info "Legacy/BIOS boot, or firmware does not expose the UEFI variable."
}

Section "TPM"

try {
    $tpm = Get-Tpm -ErrorAction Stop
    if ($tpm.TpmPresent) {
        $state = if ($tpm.TpmEnabled -and $tpm.TpmReady) { 'enabled + ready' } else { 'present, not ready' }
        Warn "TPM present ($state)."
        if ($tpm.TpmOwned) { Info "TPM is OWNED - BitLocker binds to this." }
    } else {
        Ok "No TPM reported."
    }
} catch {
    Info "Get-Tpm unavailable: $($_.Exception.Message)"
}

# ----------------------------------------------------------- Fast Startup ----
Section "Fast Startup (hiberboot)"

try {
    $hib = (Get-ItemProperty 'HKLM:\SYSTEM\CurrentControlSet\Control\Session Manager\Power' `
            -Name HiberbootEnabled -ErrorAction Stop).HiberbootEnabled
    if ($hib -eq 1) {
        Bad "Fast Startup is ENABLED - it locks NTFS and can stop Linux mounting it."
        Info "Disable with:  powercfg /h off"
    } else {
        Ok "Fast Startup is disabled."
    }
} catch {
    Info "Could not read HiberbootEnabled."
}


# ------------------------------------------------------------------ Disks ----
Section "Disks"

$EFI_GPT = '{c12a7328-f81f-11d2-ba4b-00a0c93ec93b}'
$LINUX_GPT = '{0fc63daf-8483-4772-8e79-3d69d8477de4}'

function Get-GptType($diskNumber, $partNumber) {
    try {
        (Get-Partition -DiskNumber $diskNumber -PartitionNumber $partNumber).GptType
    } catch { $null }
}

try {
    foreach ($d in (Get-Disk | Sort-Object Number)) {
        $sizeGb = [math]::Round($d.Size / 1GB, 0)
        Write-Host ("  Disk {0}: {1}  {2} GB  {3}  [{4}]" -f `
            $d.Number, $d.FriendlyName, $sizeGb, $d.PartitionStyle, $d.BusType) -ForegroundColor White

        foreach ($p in (Get-Partition -DiskNumber $d.Number -ErrorAction SilentlyContinue)) {
            $pGb = [math]::Round($p.Size / 1GB, 1)
            $letter = if ($p.DriveLetter) { "$($p.DriveLetter):" } else { "--" }
            $gpt = Get-GptType $d.Number $p.PartitionNumber
            $label = switch ($gpt) {
                $EFI_GPT   { 'EFI System Partition' }
                $LINUX_GPT { 'Linux filesystem' }
                default    { '' }
            }
            Write-Host ("      part {0}  {1,-4} {2,7} GB  {3}" -f `
                $p.PartitionNumber, $letter, $pGb, $label) -ForegroundColor Gray
        }
    }
} catch {
    Warn "Disk enumeration failed: $($_.Exception.Message)"
}

# --------------------------------------------- Existing Linux/Omarchy? -------
Section "Existing Linux / Omarchy install"

$sawLinux = $false
try {
    foreach ($d in (Get-Disk | Sort-Object Number)) {
        foreach ($p in (Get-Partition -DiskNumber $d.Number -ErrorAction SilentlyContinue)) {
            $gpt = Get-GptType $d.Number $p.PartitionNumber
            if ($gpt -eq $LINUX_GPT) {
                $sawLinux = $true
                Ok "Disk $($d.Number) part $($p.PartitionNumber) is a Linux filesystem partition ($([math]::Round($p.Size/1GB,1)) GB)."
            }
            if ($gpt -eq $EFI_GPT) {
                if (-not $p.DriveLetter) {
                    Warn "ESP on disk $($d.Number) has NO drive letter - needs elevation/dismount to inspect."
                    Info "  This is often where a second OS keeps its bootloader. Mount it to look deeper:"
                    Info "  mountvol X: /s   (elevated), then check X:\EFI"
                    continue
                }
                $efiDir = "$($p.DriveLetter):\EFI"
                if (-not (Test-Path $efiDir)) { continue }
                $dirs = @(Get-ChildItem $efiDir -Directory -ErrorAction SilentlyContinue |
                          Select-Object -ExpandProperty Name)
                Write-Host "  $efiDir -> $($dirs -join ', ')" -ForegroundColor White
                if ($dirs -match 'limine|arch|omarchy|systemd|grub|ubuntu') {
                    Ok "Non-Microsoft bootloader present - Omarchy/Arch is likely already installed."
                }
            }
        }
    }
} catch {
    Warn "Boot-entry inspection failed: $($_.Exception.Message)"
}

try {
    $bcd = & bcdedit /enum firmware 2>&1 | Out-String
    if ($bcd -match 'Linux|Arch|Omarchy|Limine|GRUB') {
        Ok "Firmware boot entries reference a Linux loader:"
        ($bcd -split "`n" | Where-Object { $_ -match 'Linux|Arch|Omarchy|Limine|GRUB' }) |
            Select-Object -First 10 | ForEach-Object { Info $_.Trim() }
    }
} catch {
    Info "bcdedit /enum firmware unavailable (needs elevation)."
}

# ------------------------------------------------------------------ Summary --
Section "Next steps"

Write-Host @"
  1. BitLocker ON?  Back up the 48-digit recovery key FIRST.
     Only this step is irreversible if skipped.

  2. Omarchy already in the BIOS boot menu?  Then you are done -
     pick it at boot. No reinstall needed.

  3. Omarchy on a separate physical disk (you have 3 NVMe drives)?
     Disconnect the Windows disk's power before running the installer.
     Highest-confidence way to leave Windows' bootloader untouched.

  4. Disable Fast Startup:  powercfg /h off

  5. Expect the Omarchy installer to require Secure Boot / TPM disabled.
"@ -ForegroundColor Gray

if (-not $sawLinux) {
    Write-Host ""
    Warn "No Linux filesystem partition found. Omarchy may be on a disk that is not attached,"
    Info "or not installed yet."
}

}
