function New-PyVenv {
    param (
        $name = 'py_venv'
    )

    "Clear git untracked files" | Write-Host -BackgroundColor black -ForegroundColor Green
    git clean -xfd
    "Create python virtual environment, named $name" | Write-Host -BackgroundColor black -ForegroundColor Green
    py -3 -m venv $name
}

function Enter-PyVenv {
    param (
      $name = 'py_venv'
    )

    "Enter python virtual environment" | Write-Host -BackgroundColor black -ForegroundColor Green
    & "$name\Scripts\activate.ps1"
}

function Exit-PyVenv {
    "Exit python virtual environment" | Write-Host -BackgroundColor black -ForegroundColor Green
    deactivate
}

function Install-PyVenvPackages {
    param (
        $name = 'py_venv'
    )

    $pip_packages_host = 'https://pypi.tuna.tsinghua.edu.cn/simple'

    "Install web packages......" | Write-Host -BackgroundColor black -ForegroundColor Green
    & "$name\Scripts\python" -m pip install -U pip wheel setuptools pyreadline==2.1 autopep8 -i ($pip_packages_host)
    & "$name\Scripts\python" -m pip install -U librosa ipykernel opencv-python -i ($pip_packages_host)
    & "$name\Scripts\python" -m pip install -U paddlepaddle -i https://mirror.baidu.com/pypi/simple
}
