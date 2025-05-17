# Create a scheduled task to run the backup script daily
$action = New-ScheduledTaskAction -Execute "powershell.exe" -Argument "-File `"$PSScriptRoot\backup_dashboards.sh`""
$trigger = New-ScheduledTaskTrigger -Daily -At 2AM
$settings = New-ScheduledTaskSettingsSet -StartWhenAvailable -DontStopOnIdleEnd

Register-ScheduledTask -TaskName "GrafanaDashboardBackup" -Action $action -Trigger $trigger -Settings $settings -Description "Daily backup of Grafana dashboards" 