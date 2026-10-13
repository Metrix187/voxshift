; VoxShift installer (Inno Setup 6).
; Built by packaging/build.py, which passes AppVersion / DistDir / OutDir.
; Per-user install (%LOCALAPPDATA%\Programs), so the app itself needs no admin rights;
; only the optional virtual-microphone driver asks for elevation.

#ifndef AppVersion
  #define AppVersion "0.0.0"
#endif
#ifndef DistDir
  #define DistDir "..\build\dist\VoxShift"
#endif
#ifndef OutDir
  #define OutDir "..\build"
#endif

[Setup]
AppId={{80A8D700-17F2-43C6-8A8B-6618128A3C70}
AppName=VoxShift
AppVersion={#AppVersion}
AppVerName=VoxShift {#AppVersion}
AppPublisher=VoxShift
DefaultDirName={localappdata}\Programs\VoxShift
DefaultGroupName=VoxShift
DisableProgramGroupPage=yes
PrivilegesRequired=lowest
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
OutputDir={#OutDir}
OutputBaseFilename=VoxShift-Setup-{#AppVersion}
SetupIconFile=..\voxshift\assets\voxshift.ico
UninstallDisplayIcon={app}\VoxShift.exe
UninstallDisplayName=VoxShift
WizardStyle=modern
; model weights are already dense; fast LZMA keeps build time sane for ~1 GB
Compression=lzma2/fast
SolidCompression=no
CloseApplications=yes
RestartApplications=no

[Tasks]
Name: "vbcable"; Description: "Install the virtual microphone (free VB-CABLE driver from vb-audio.com; Windows will ask for permission)"; Check: not CableInstalled
Name: "desktopicon"; Description: "Create a desktop shortcut"
Name: "autostart"; Description: "Start VoxShift with Windows (in the tray)"; Flags: unchecked

[Files]
Source: "{#DistDir}\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{autoprograms}\VoxShift"; Filename: "{app}\VoxShift.exe"
Name: "{autodesktop}\VoxShift"; Filename: "{app}\VoxShift.exe"; Tasks: desktopicon

[Registry]
Root: HKCU; Subkey: "Software\Microsoft\Windows\CurrentVersion\Run"; ValueType: string; ValueName: "VoxShift"; ValueData: """{app}\VoxShift.exe"" --minimized"; Flags: uninsdeletevalue; Tasks: autostart

[Run]
; 1) download the official VB-CABLE pack and refuse it unless its Authenticode signature is VB-Audio's;
; 2) Setup itself runs the verified driver setup elevated (-i -h = install, hidden UI). Requesting elevation
;    from this visible wizard keeps the UAC prompt in front; Windows also asks for driver consent once.
Filename: "{app}\voxshift-cli.exe"; Parameters: "cable --download ""{app}\vbcable"""; StatusMsg: "Downloading the virtual microphone driver (VB-CABLE)..."; Flags: runhidden waituntilterminated; Tasks: vbcable
Filename: "{app}\vbcable\VBCABLE_Setup_x64.exe"; Parameters: "-i -h"; WorkingDir: "{app}\vbcable"; StatusMsg: "Installing the virtual microphone - approve the Windows prompts..."; Verb: runas; Flags: shellexec waituntilterminated; Tasks: vbcable; Check: VBCableDownloaded
Filename: "{app}\VoxShift.exe"; Description: "Launch VoxShift"; Flags: nowait postinstall skipifsilent

[UninstallDelete]
Type: filesandordirs; Name: "{app}\_internal"
Type: filesandordirs; Name: "{app}\vbcable"

[Code]
function VBCableDownloaded: Boolean;
begin
  Result := FileExists(ExpandConstant('{app}\vbcable\VBCABLE_Setup_x64.exe'));
end;

function CableInstalled: Boolean;
begin
  { The driver runs from the DriverStore, so look for its service, not a file in System32\drivers. }
  Result := RegKeyExists(HKLM, 'SYSTEM\CurrentControlSet\Services\VBAudioVACMME');
end;
