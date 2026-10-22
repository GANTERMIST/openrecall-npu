; Установщик WinAI (Inno Setup 6). Собирается в GitHub Actions:
;   iscc /DAppVersion=1.0.0 installer\winai.iss
; Ставится в профиль пользователя, права администратора не нужны.

#ifndef AppVersion
  #define AppVersion "0.0.0-dev"
#endif

[Setup]
AppId={{7C1F3E8A-5B2D-4A9E-9C61-2E8B7D4F0A13}
AppName=WinAI
AppVersion={#AppVersion}
AppPublisher=GANTERMIST
AppPublisherURL=https://github.com/GANTERMIST/openrecall-npu
DefaultDirName={localappdata}\Programs\WinAI
DefaultGroupName=WinAI
DisableProgramGroupPage=yes
PrivilegesRequired=lowest
OutputDir=..\dist
OutputBaseFilename=WinAI-Setup-{#AppVersion}
SetupIconFile=winai.ico
UninstallDisplayIcon={app}\WinAI.exe
Compression=lzma2
SolidCompression=yes
WizardStyle=modern
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
CloseApplications=yes

[Languages]
Name: "ru"; MessagesFile: "compiler:Languages\Russian.isl"
Name: "en"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "Значок на рабочем столе"; GroupDescription: "Дополнительно:"
Name: "autostart"; Description: "Запускать WinAI при входе в Windows"; GroupDescription: "Дополнительно:"
Name: "models"; Description: "Скачать модели сразу (несколько ГБ, иначе — при первом использовании)"; GroupDescription: "Дополнительно:"; Flags: unchecked

[Files]
Source: "..\dist\WinAI.exe"; DestDir: "{app}"; Flags: ignoreversion
Source: "..\winai\*"; DestDir: "{app}\winai"; Excludes: "__pycache__,*.pyc"; Flags: ignoreversion recursesubdirs createallsubdirs
Source: "..\requirements.txt"; DestDir: "{app}"; Flags: ignoreversion
Source: "..\recall.py"; DestDir: "{app}"; Flags: ignoreversion
Source: "..\README.md"; DestDir: "{app}"; Flags: ignoreversion isreadme
Source: "..\config.example.toml"; DestDir: "{app}"; Flags: ignoreversion

[Icons]
Name: "{group}\WinAI"; Filename: "{app}\WinAI.exe"
Name: "{group}\WinAI — переустановить зависимости"; Filename: "{app}\WinAI.exe"; Parameters: "--setup"
Name: "{group}\Удалить WinAI"; Filename: "{uninstallexe}"
Name: "{userdesktop}\WinAI"; Filename: "{app}\WinAI.exe"; Tasks: desktopicon
Name: "{userstartup}\WinAI"; Filename: "{app}\WinAI.exe"; Tasks: autostart

[Run]
Filename: "{app}\WinAI.exe"; Parameters: "--models"; Description: "Подготовить и запустить WinAI"; Tasks: models; Flags: postinstall nowait
Filename: "{app}\WinAI.exe"; Description: "Подготовить и запустить WinAI"; Tasks: not models; Flags: postinstall nowait

[UninstallRun]
; остановить запущенный WinAI (только процессы Python из папки программы)
Filename: "powershell.exe"; Parameters: "-NoProfile -Command ""Get-Process python,pythonw -ErrorAction SilentlyContinue | Where-Object {{ $_.Path -like '{app}\*' }} | Stop-Process -Force"""; Flags: runhidden; RunOnceId: "StopWinAI"

[UninstallDelete]
; окружение и скачанные инструменты; ваши данные (data) и модели (models) остаются
Type: filesandordirs; Name: "{app}\.venv"
Type: filesandordirs; Name: "{app}\tools"
Type: filesandordirs; Name: "{app}\winai"
