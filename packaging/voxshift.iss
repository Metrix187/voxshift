; VoxShift installer (Inno Setup 6).
; Built by packaging/build.py, which passes AppVersion / DistDir / OutDir and, for the web installer,
; ModelInclude (generated Pascal that lists the AI voice pack chunks with their SHA-256).
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
AppPublisherURL=https://github.com/Metrix187/voxshift
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
Compression=lzma2/ultra64
SolidCompression=yes
CloseApplications=yes
RestartApplications=no

[Tasks]
#ifdef ModelInclude
Name: "aivoices"; Description: "Download AI voices now (about 820 MB; you can also do this later in the app)"; Check: not ModelInstalled
#endif
Name: "vbcable"; Description: "Install the virtual microphone (free VB-CABLE driver from vb-audio.com; Windows will ask for permission)"; Check: not CableInstalled
Name: "desktopicon"; Description: "Create a desktop shortcut"
Name: "autostart"; Description: "Start VoxShift with Windows (in the tray)"; Flags: unchecked

[InstallDelete]
; an earlier offline (bundled-model) install kept the model here; web installs use %LOCALAPPDATA%
Type: filesandordirs; Name: "{app}\models"

[Files]
Source: "{#DistDir}\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{autoprograms}\VoxShift"; Filename: "{app}\VoxShift.exe"
Name: "{autodesktop}\VoxShift"; Filename: "{app}\VoxShift.exe"; Tasks: desktopicon

[Registry]
Root: HKCU; Subkey: "Software\Microsoft\Windows\CurrentVersion\Run"; ValueType: string; ValueName: "VoxShift"; ValueData: """{app}\VoxShift.exe"" --minimized"; Flags: uninsdeletevalue; Tasks: autostart

[Run]
#ifdef ModelInclude
; the chunks were downloaded and hash-checked on the "Preparing" page; rebuild + verify the model files
Filename: "{app}\voxshift-cli.exe"; Parameters: "models assemble --chunks ""{tmp}"""; StatusMsg: "Installing AI voices..."; Flags: runhidden waituntilterminated; Tasks: aivoices
#endif
; 1) download the official VB-CABLE pack and refuse it unless its Authenticode signature is VB-Audio's;
; 2) Setup itself runs the verified driver setup elevated (-i -h = install, hidden UI). Requesting elevation
;    from this visible wizard keeps the UAC prompt in front; Windows also asks for driver consent once.
Filename: "{app}\voxshift-cli.exe"; Parameters: "cable --download ""{app}\vbcable"""; StatusMsg: "Downloading the virtual microphone driver (VB-CABLE)..."; Flags: runhidden waituntilterminated; Tasks: vbcable
Filename: "{app}\vbcable\VBCABLE_Setup_x64.exe"; Parameters: "-i -h"; WorkingDir: "{app}\vbcable"; StatusMsg: "Installing the virtual microphone - approve the Windows prompts..."; Verb: runas; Flags: shellexec waituntilterminated; Tasks: vbcable; Check: VBCableDownloaded
Filename: "{app}\VoxShift.exe"; Description: "Launch VoxShift"; Flags: nowait postinstall skipifsilent

[UninstallDelete]
Type: filesandordirs; Name: "{app}\_internal"
Type: filesandordirs; Name: "{app}\vbcable"
; downloaded AI voice pack (settings and your own voices in %APPDATA%\VoxShift are kept)
Type: filesandordirs; Name: "{localappdata}\VoxShift\models"

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

function ModelInstalled: Boolean;
begin
  Result := FileExists(ExpandConstant('{localappdata}\VoxShift\models\onnx\whisper-small-bigvgan\config.json'));
end;

{ Ask a running VoxShift to exit before its files are replaced. It hides in the tray when its
  window is closed, so the Restart Manager alone cannot shut it down. }
function OpenEvent(Access: Cardinal; Inherit: LongBool; Name: String): THandle;
  external 'OpenEventW@kernel32.dll stdcall';
function SetEvent(Handle: THandle): LongBool;
  external 'SetEvent@kernel32.dll stdcall';
function CloseHandle(Handle: THandle): LongBool;
  external 'CloseHandle@kernel32.dll stdcall';

function PrepareToInstall(var NeedsRestart: Boolean): String;
var
  H: THandle;
  I: Integer;
begin
  Result := '';
  H := OpenEvent($0002, False, 'Local\VoxShift.Quit');
  if H <> 0 then begin
    SetEvent(H);
    CloseHandle(H);
    for I := 1 to 50 do begin  { the event vanishes once the app has exited; wait up to 5 s }
      Sleep(100);
      H := OpenEvent($0002, False, 'Local\VoxShift.Quit');
      if H = 0 then Break;
      CloseHandle(H);
    end;
  end;
end;

#ifdef ModelInclude
var
  DownloadPage: TDownloadWizardPage;

{ Queue a file unless an earlier attempt already left a verified copy in Setup's temp folder,
  so pressing Install again after a network error only fetches what is still missing. }
procedure AddChunk(const Url, Name, Sha: String);
var
  Path: String;
begin
  Path := ExpandConstant('{tmp}\' + Name);
  if FileExists(Path) and (GetSHA256OfFile(Path) = Sha) then
    Log('already have ' + Name)
  else
    DownloadPage.Add(Url, Name, Sha);
end;

#include ModelInclude

function OnDownloadProgress(const Url, FileName: String; const Progress, ProgressMax: Int64): Boolean;
begin
  Result := True;
end;

procedure InitializeWizard;
begin
  DownloadPage := CreateDownloadPage('Downloading AI voices',
    'VoxShift is downloading its AI voice pack (about 820 MB). Every piece is checked before use.',
    @OnDownloadProgress);
end;

function NextButtonClick(CurPageID: Integer): Boolean;
begin
  Result := True;
  if (CurPageID = wpReady) and WizardIsTaskSelected('aivoices') then begin
    DownloadPage.Clear;
    AddModelDownloads;
    DownloadPage.Show;
    try
      try
        DownloadPage.Download;
      except
        if DownloadPage.AbortedByUser then
          Log('AI voice download aborted by user.')
        else
          { (no line may start with '#': the preprocessor would read it as a directive) }
          SuppressibleMsgBox('The AI voice download was interrupted: ' + GetExceptionMessage + #13#10#13#10 +
            'Click Install to resume, or untick "Download AI voices" to skip it ' +
            '(VoxShift can download them later).', mbError, MB_OK, IDOK);
        Result := False;
      end;
    finally
      DownloadPage.Hide;
    end;
  end;
end;
#endif
