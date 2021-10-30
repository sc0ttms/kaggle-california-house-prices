# common params
$PY_VENV_NAME = 'py_venv'

# import powershell modules
$PS_MODULES_PATH = '.\PS-Modules\'
$fileList = Get-ChildItem $PS_MODULES_PATH -Recurse *.psd1
foreach ($file in $fileList) {
    $ps_module_name = $file.BaseName
    if (Get-Module -Name $ps_module_name) {
        # Write-Host $ps_module_name
    }
    else {
        Import-Module $file
    }
}

# config run mode
"Select run mode" | Write-Host -BackgroundColor black -ForegroundColor Green 
"1. Update PS-Modules
2. Create Environment" | Write-Host -BackgroundColor black -ForegroundColor Yellow
$mode = Read-Host "Enter number"

switch ($mode) {
    {$mode -eq "1"} { 
        foreach ($file in $fileList) {
            $ps_module_name = $file.BaseName
            if (Get-Module -Name $ps_module_name) {
                Remove-Module $ps_module_name
                "Remove Module $ps_module_name" | Write-Host -BackgroundColor black -ForegroundColor Yellow
            }
            Import-Module $file
            "Update Module $ps_module_name" | Write-Host -BackgroundColor black -ForegroundColor Yellow
        }
    }
    {$mode -eq "2"} { 
        New-PyVenv -name $PY_VENV_NAME
        Install-PyVenvPackages -name $PY_VENV_NAME
    }
    Default {
        "Error: not support mode!" | Write-Host -BackgroundColor black -ForegroundColor Red
    }
}
