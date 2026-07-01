$secpass = ConvertTo-SecureString 'yxyyxy' -AsPlainText -Force
$cred = New-Object System.Management.Automation.PSCredential('pi', $secpass)

# Try using ssh with password via command
$cmd = @"
ssh -o StrictHostKeyChecking=no pi@10.68.243.28 "lsusb; echo ---; ls -l /dev/ttyUSB* 2>/dev/null; echo ---; dmesg | grep -i ch34 | tail -5"
"@
Write-Host $cmd
