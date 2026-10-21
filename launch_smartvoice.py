"""SmartVoiceStudio 一键启动器。

先启动 CosyVoice3 本机服务，等待 /health 成功，再打开桌面 GUI。
GUI 关闭后，只停止本启动器创建的 CosyVoice3 服务；不影响原本已在运行的服务。
默认路径按开发电脑配置，可使用环境变量覆盖。
"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen


# 路径配置：其他电脑部署时修改这里，或使用相同名称的环境变量
COSYVOICE_PYTHON = Path(os.getenv("COSYVOICE_PYTHON", r"D:\Anaconda\envs_dir\cosyvoice3\python.exe"))
SMARTVOICE_PYTHON = Path(os.getenv("SMARTVOICE_PYTHON", r"D:\Anaconda\envs_dir\SmartVoiceStudio\python.exe"))
COSYVOICE_REPO = Path(os.getenv("COSYVOICE_REPO", r"E:\CosyVoiceRuntime\CosyVoice"))
COSYVOICE_MODEL = Path(os.getenv("COSYVOICE_MODEL", r"E:\CosyVoiceRuntime\CosyVoice\pretrained_models\Fun-CosyVoice3-0.5B"))
HOST = "127.0.0.1"
PORT = 8765
START_TIMEOUT = 480  # Intel Arc 电脑当前按 CPU 模式运行，预留模型加载时间


def find_project():
    # 普通脚本或 PyInstaller 单文件 EXE 都能从工程附近找到 main.py
    locations = [Path.cwd(), Path(__file__).resolve().parent, Path(sys.executable).resolve().parent]
    locations += [parent for path in list(locations) for parent in path.parents]
    custom = os.getenv("SMARTVOICE_PROJECT")
    if custom:
        locations.insert(0, Path(custom))
    for path in locations:
        if (path / "main.py").is_file() and (path / "tts" / "local_server.py").is_file():
            return path.resolve()
    raise FileNotFoundError("找不到 SmartVoiceStudio 项目根目录，请把启动器放在项目内，或设置 SMARTVOICE_PROJECT")


def conda_environment(python_path):
    """为直接调用 Conda 环境解释器补齐常见 Windows DLL 搜索路径。"""
    env = os.environ.copy()
    prefix = python_path.parent
    paths = [prefix, prefix / "Scripts", prefix / "Library" / "bin", prefix / "Library" / "usr" / "bin", prefix / "Library" / "mingw-w64" / "bin"]
    env["PATH"] = os.pathsep.join(str(path) for path in paths) + os.pathsep + env.get("PATH", "")
    env["CONDA_PREFIX"] = str(prefix)
    env["CONDA_DEFAULT_ENV"] = prefix.name
    env["PYTHONUNBUFFERED"] = "1"
    return env


def check_health():
    """只接受本机的 CosyVoice3 服务，避免把其他程序误认为 TTS。"""
    try:
        with urlopen(f"http://{HOST}:{PORT}/health", timeout=1.5) as response:
            info = json.load(response)
        return isinstance(info, dict) and info.get("ok") is True and info.get("model") == "CosyVoice3"
    except (OSError, ValueError, URLError):
        return False


def show_log_tail(log_path, count=35):
    if not log_path.is_file():
        return
    data = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
    print("\nCosyVoice 服务日志（最后几行）：")
    for line in data[-count:]:
        print(line)


def stop_process(process):
    if process is None or process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=15)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=10)


def main():
    parser = argparse.ArgumentParser(description="一键启动 CosyVoice3 和 SmartVoiceStudio")
    parser.add_argument("--source", action="store_true", help="强制使用 Python main.py，而不是已打包的 GUI EXE")
    parser.add_argument("--doctor", action="store_true", help="只检查路径和当前服务状态，不启动任何程序")
    args = parser.parse_args()

    project = find_project()
    gui_exe = project / "dist" / "SmartVoiceStudio" / "SmartVoiceStudio.exe"
    use_exe = gui_exe.is_file() and not args.source
    gui_python_ok = SMARTVOICE_PYTHON.is_file()
    checks = {
        "GUI 源码": project / "main.py",
        "本机 TTS 服务源码": project / "tts" / "local_server.py",
        "CosyVoice 环境 Python": COSYVOICE_PYTHON,
        "CosyVoice 官方源码": COSYVOICE_REPO / "cosyvoice" / "cli" / "cosyvoice.py",
        "CosyVoice 模型配置": COSYVOICE_MODEL / "cosyvoice3.yaml",
    }
    if not use_exe:
        checks["SmartVoiceStudio 环境 Python"] = SMARTVOICE_PYTHON

    print(f"工程位置：{project}")
    print(f"GUI 启动方式：{'PyInstaller EXE' if use_exe else 'Python 源码'}")
    for title, path in checks.items():
        print(f"[{'OK' if path.is_file() else '缺失'}] {title}: {path}")
    print(f"本机 TTS 服务：{'已就绪' if check_health() else '未就绪'}")

    if args.doctor:
        return 0 if all(p.is_file() for p in checks.values()) else 1
    missing = [(title, path) for title, path in checks.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError("必要文件不存在，请先检查以上缺失的路径")

    log_dir = project / "runtime_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    server_log = log_dir / "cosyvoice_server.log"
    server_process = None
    gui_process = None

    try:
        if check_health():
            print("检测到已经运行的 CosyVoice3 服务，直接复用。")
        else:
            print("正在启动 CosyVoice3，请等待模型加载（CPU 模式可能耗时数分钟）...")
            command = [str(COSYVOICE_PYTHON), "-u", "-m", "tts.local_server", "--repo", str(COSYVOICE_REPO), "--model", str(COSYVOICE_MODEL), "--port", str(PORT)]
            with server_log.open("w", encoding="utf-8") as log_file:
                flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
                server_process = subprocess.Popen(command, cwd=str(project), env=conda_environment(COSYVOICE_PYTHON), stdout=log_file, stderr=subprocess.STDOUT, creationflags=flags)

            start = time.monotonic()
            next_message = start
            while not check_health():
                if server_process.poll() is not None:
                    show_log_tail(server_log)
                    raise RuntimeError(f"CosyVoice3 服务异常退出，退出码：{server_process.returncode}")
                elapsed = time.monotonic() - start
                if elapsed > START_TIMEOUT:
                    show_log_tail(server_log)
                    raise TimeoutError(f"等待 TTS 服务超过 {START_TIMEOUT} 秒。完整日志：{server_log}")
                if time.monotonic() >= next_message:
                    print(f"模型加载中... 已等待 {int(elapsed)} 秒")
                    next_message = time.monotonic() + 8
                time.sleep(1.0)
            print("CosyVoice3 本地服务已就绪。")

        if use_exe:
            cmd = [str(gui_exe)]
            env = os.environ.copy()
        else:
            cmd = [str(SMARTVOICE_PYTHON), str(project / "main.py")]
            env = conda_environment(SMARTVOICE_PYTHON)
        # GUI 与源码模式共用项目根目录的 results 和录音文件夹，防止重打包清空音频
        env["SMARTVOICE_DATA_HOME"] = str(project)
        print("正在启动 SmartVoiceStudio GUI...")
        gui_process = subprocess.Popen(cmd, cwd=str(project), env=env)
        gui_code = gui_process.wait()
        print(f"SmartVoiceStudio 已关闭，GUI 退出码：{gui_code}")
        return gui_code
    finally:
        if gui_process is not None and gui_process.poll() is None:
            stop_process(gui_process)
        if server_process is not None:
            print("正在关闭本次启动的 CosyVoice3 服务...")
            stop_process(server_process)
        else:
            print("未关闭原本已存在的 CosyVoice3 服务。")


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (FileNotFoundError, TimeoutError, RuntimeError, OSError) as exc:
        print(f"\n启动失败：{exc}", file=sys.stderr)
        sys.exit(1)
