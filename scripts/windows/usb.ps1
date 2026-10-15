# USB selection is stored with the VM; QEMU owns reconnect polling.
function Get-ReimsUsbSelections($Config) {
    if (!$Config.PSObject.Properties['usb_devices']) { return }
    $ids = [Collections.Generic.HashSet[string]]::new()
    foreach ($device in @($Config.usb_devices)) {
        if ($device.bus -lt 1 -or $device.bus -gt 65535 -or
            $device.port -cnotmatch '^[1-9][0-9]*(\.[1-9][0-9]*){0,6}$' -or
            $device.vendor -cnotmatch '^[0-9a-f]{4}$' -or $device.vendor -eq '0000' -or
            $device.product -cnotmatch '^[0-9a-f]{4}$' -or $device.product -eq '0000') {
            throw 'Invalid saved USB selection.'
        }
        $expected = "hostusb-$($device.bus)-$($device.port.Replace('.','_'))-$($device.vendor)$($device.product)"
        if ($device.id -cne $expected -or !$ids.Add($expected)) { throw 'Invalid or duplicate USB selection ID.' }
        $device
    }
}

function Get-ReimsUsbArgument($Device) {
    # shortcut: selection follows a physical port, reselect after moving ports.
    return @{driver='usb-host'; id=$Device.id; bus='xhci.0'; hostbus=[int]$Device.bus;
        hostport=[string]$Device.port; vendorid=[Convert]::ToInt32($Device.vendor,16);
        productid=[Convert]::ToInt32($Device.product,16); 'guest-reset'=$false}
}

function Remove-ReimsUsbLaunchArguments([string[]]$Arguments) {
    for ($i=0; $i -lt $Arguments.Count; $i++) {
        if ($Arguments[$i] -eq '-device' -and $i+1 -lt $Arguments.Count -and $Arguments[$i+1].StartsWith('{')) {
            $device=$Arguments[$i+1] | ConvertFrom-Json
            if ($device.driver -eq 'usb-host') { $i++; continue }
        }
        $Arguments[$i]
    }
}

function Get-ReimsPnpProperties([string[]]$Instances, [string]$Key) {
    # WMI rejects large OR queries, so batch within its quota.
    for ($offset=0; $offset -lt $Instances.Count; $offset+=24) {
        Get-PnpDeviceProperty -InstanceId $Instances[$offset..([math]::Min($offset+23,$Instances.Count-1))] -KeyName $Key -ErrorAction Stop
    }
}

function Get-ReimsProtectedUsbIds {
    $blocked = [Collections.Generic.HashSet[string]]::new([StringComparer]::OrdinalIgnoreCase)
    $storageRoots = @()
    $protected = @(Get-PnpDevice -PresentOnly -Class Keyboard,Mouse,Bluetooth -ErrorAction SilentlyContinue | ForEach-Object InstanceId)
    foreach ($drive in @($ProjectRoot,$VmRoot,$env:SystemRoot) | ForEach-Object { [IO.Path]::GetPathRoot($_).TrimEnd('\',':') } | Select-Object -Unique) {
        if ($drive -match '^[A-Za-z]$') {
            foreach ($partition in Get-Partition -DriveLetter $drive -ErrorAction Stop) {
                $disks = @(Get-CimInstance Win32_DiskDrive -Filter "Index=$($partition.DiskNumber)" -ErrorAction Stop)
                if (!$disks.Count) { throw 'Cannot identify the host storage device; USB forwarding was not changed.' }
                $protected += $disks | ForEach-Object PNPDeviceID
                if ((Get-Disk -Number $partition.DiskNumber -ErrorAction Stop).BusType -eq 'USB') { $storageRoots += $disks | ForEach-Object PNPDeviceID }
            }
        }
    }
    $visited = [Collections.Generic.HashSet[string]]::new([StringComparer]::OrdinalIgnoreCase)
    $parents = @{}
    for ($depth=0; $protected.Count -and $depth -lt 32; $depth++) {
        $pending = @($protected | Where-Object { $_ -and $visited.Add($_) })
        foreach ($instance in $pending) {
            if ($instance -match 'VID_([0-9A-F]{4})&PID_([0-9A-F]{4})') { [void]$blocked.Add("$($Matches[1]):$($Matches[2])") }
        }
        $properties = @(Get-ReimsPnpProperties $pending DEVPKEY_Device_Parent | Where-Object { $_.PSObject.Properties['Data'] })
        foreach ($property in $properties) { $parents[$property.InstanceId]=$property.Data }
        $protected = @($properties | ForEach-Object Data)
    }
    foreach ($instance in $storageRoots) {
        $identified=$false
        for ($depth=0; $instance -and $depth -lt 32; $depth++) {
            if ($instance -match 'VID_([0-9A-F]{4})&PID_([0-9A-F]{4})') { $identified=$true; break }
            $instance=$parents[$instance]
        }
        if (!$identified) { throw 'USB 保存先の識別情報を取得できません。転送は開始しません。' }
    }
    return ,$blocked
}

function Assert-ReimsUsbSafe($Selections) {
    if (!@($Selections).Count) { return }
    $blocked = Get-ReimsProtectedUsbIds
    foreach ($device in $Selections) {
        if ($blocked.Contains("$($device.vendor):$($device.product)")) {
            throw 'Windows の入力機器または VM の保存先ディスクは転送できません。USB の登録を解除してください。'
        }
    }
}

function ConvertFrom-ReimsUsbInventory([string]$Text, $Names, $Blocked) {
    foreach ($match in [regex]::Matches($Text, '(?im)^\s*Bus (\d+), Addr (\d+), Port ([0-9.]+),[^\r\n]*\r?\n\s*Class [0-9a-f]+: USB device ([0-9a-f]{4}):([0-9a-f]{4})')) {
        $bus=[int]$match.Groups[1].Value; $port=$match.Groups[3].Value
        $vendor=$match.Groups[4].Value.ToLowerInvariant(); $product=$match.Groups[5].Value.ToLowerInvariant()
        $key="${vendor}:${product}"
        if ($Blocked.Contains($key)) { continue }
        [pscustomobject]@{id="hostusb-$bus-$($port.Replace('.','_'))-$vendor$product";
            bus=$bus; address=[int]$match.Groups[2].Value; port=$port; vendor=$vendor; product=$product;
            label="$key $($Names[$key]) — USB $bus / $port"; present=$true; selected=$false}
    }
}

function Get-ReimsUsbDevices {
    $config = Get-ReimsVmConfig
    $saved = @(Get-ReimsUsbSelections $config)
    $runtime = Get-ReimsRuntime
    $devices = @()
    if ($runtime -and (Get-ReimsVmProcess $runtime)) {
        $text = Invoke-ReimsQmp $runtime human-monitor-command @{'command-line'='info usbhost'}
        if ($text -match 'not available|unknown command') { throw 'USB 転送に対応した QEMU へ更新してください。' }
        $names = @{}
        $pnp = @(Get-PnpDevice -PresentOnly | Where-Object InstanceId -Like 'USB\VID_*')
        $descriptions = @{}
        foreach ($property in Get-ReimsPnpProperties @($pnp | ForEach-Object InstanceId) DEVPKEY_Device_BusReportedDeviceDesc) {
            if ($property.PSObject.Properties['Data']) { $descriptions[$property.InstanceId]=$property.Data }
        }
        foreach ($device in $pnp) {
            if ($device.InstanceId -match 'VID_([0-9A-F]{4})&PID_([0-9A-F]{4})') {
                $key="$($Matches[1]):$($Matches[2])"
                $description=$descriptions[$device.InstanceId]
                $names[$key]=if ($description) { $description } else { $device.FriendlyName }
            }
        }
        $devices = @(ConvertFrom-ReimsUsbInventory $text $names (Get-ReimsProtectedUsbIds))
    }
    foreach ($device in $saved) {
        $row = $devices | Where-Object id -EQ $device.id | Select-Object -First 1
        if ($row) { $row.selected=$true; $row.label += ' ［自動接続］' }
        else {
            $devices += [pscustomobject]@{id=$device.id;bus=$device.bus;address=0;port=$device.port;
                vendor=$device.vendor;product=$device.product;present=$false;selected=$true;
                label="$($device.label) ［登録済み・接続待ち］"}
        }
    }
    return $devices
}

function Set-ReimsUsbSelection([string]$Id, [bool]$Connect) {
    $config=Get-ReimsVmConfig
    $saved=@(Get-ReimsUsbSelections $config)
    $existing=$saved | Where-Object id -EQ $Id | Select-Object -First 1
    $runtime=Get-ReimsRuntime
    $running=$runtime -and (Get-ReimsVmProcess $runtime)
    if ($Connect) {
        if ($existing) { 'この USB は自動接続に登録済みです。'; return }
        $device=Get-ReimsUsbDevices | Where-Object { $_.id -eq $Id -and $_.present } | Select-Object -First 1
        if (!$device) { throw '選択した USB がありません。一覧を更新してください。' }
        Assert-ReimsUsbSafe @($device)
        $entry=[pscustomobject]@{id=$device.id;bus=$device.bus;port=$device.port;vendor=$device.vendor;product=$device.product;label=$device.label}
        Invoke-ReimsQmp $runtime device_add (Get-ReimsUsbArgument $entry) | Out-Null
        $config | Add-Member -Force usb_devices @($saved + $entry)
        try { Write-ReimsJson (Join-Path $VmRoot 'state/config.json') $config }
        catch { Invoke-ReimsQmp $runtime device_del @{id=$Id} | Out-Null; throw }
        '自動接続に登録しました。同じ USB ポートへの再接続と VM の次回起動にも適用します。'
    } else {
        if (!$existing) { throw 'この USB は登録されていません。' }
        if ($running) {
            $objects=Invoke-ReimsQmp $runtime qom-list @{path='/machine/peripheral'}
            if ($objects | Where-Object name -EQ $Id) { Invoke-ReimsQmp $runtime device_del @{id=$Id} | Out-Null }
        }
        $config | Add-Member -Force usb_devices @($saved | Where-Object id -NE $Id)
        Write-ReimsJson (Join-Path $VmRoot 'state/config.json') $config
        'USB の自動接続を解除しました。Windows に返却します。'
    }
}
