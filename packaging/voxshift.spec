# PyInstaller spec: one folder, two executables sharing one runtime.
#   VoxShift.exe      windowed desktop app
#   voxshift-cli.exe  console tools (convert / live / cable / voices)
# Build with:  python packaging/build.py
import os

from PyInstaller.utils.hooks import collect_dynamic_libs

ROOT = os.path.abspath(os.path.join(SPECPATH, ".."))
ICON = os.path.join(ROOT, "voxshift", "assets", "voxshift.ico")

# Dev-only stacks: the standalone app runs AI voices through ONNX Runtime + numpy.
EXCLUDES = [
    "torch", "torchaudio", "torchvision", "transformers", "tokenizers", "safetensors", "huggingface_hub",
    "librosa", "numba", "llvmlite", "matplotlib", "onnx", "onnxconverter_common", "onnxruntime.transformers",
    "onnxruntime.quantization", "onnxruntime.tools", "pytest", "IPython", "pandas", "sympy", "networkx",
    "jinja2", "munch", "einops", "yaml", "tqdm", "PyInstaller",
]

a = Analysis(
    [os.path.join(SPECPATH, "entry_gui.py"), os.path.join(SPECPATH, "entry_cli.py")],
    pathex=[ROOT],
    binaries=collect_dynamic_libs("onnxruntime"),  # onnxruntime.dll + DirectML.dll
    datas=[
        (os.path.join(ROOT, "voxshift", "voices", "*.json"), "voxshift/voices"),
        (os.path.join(ROOT, "voxshift", "assets", "*"), "voxshift/assets"),
    ],
    hiddenimports=["pystray._win32", "PIL._tkinter_finder"],
    excludes=EXCLUDES,
    noarchive=False,
)
pyz = PYZ(a.pure)


def scripts_without(name):
    return [s for s in a.scripts if s[0] != name]


gui = EXE(pyz, scripts_without("entry_cli"), [], exclude_binaries=True, name="VoxShift",
          console=False, icon=ICON, upx=False)
cli = EXE(pyz, scripts_without("entry_gui"), [], exclude_binaries=True, name="voxshift-cli",
          console=True, icon=ICON, upx=False)
coll = COLLECT(gui, cli, a.binaries, a.datas, name="VoxShift", upx=False)
