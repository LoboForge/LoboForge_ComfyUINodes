import os
import shutil
import subprocess


def pick_path(title, directory):
    if not os.environ.get("DISPLAY") and not os.environ.get("WAYLAND_DISPLAY"):
        return {"error": "No display is available. Type the path into the field."}

    launchers = []
    zenity = shutil.which("zenity")
    if zenity:
        command = [zenity, "--file-selection", f"--title={title}"]
        if directory:
            command.append("--directory")
        else:
            command.extend(["--file-filter", "SQLite (*.db, *.sqlite, *.sqlite3) | *.db *.sqlite *.sqlite3"])
        launchers.append(command)
    kdialog = shutil.which("kdialog")
    if kdialog:
        if directory:
            launchers.append([kdialog, "--getexistingdirectory", os.path.expanduser("~")])
        else:
            launchers.append([kdialog, "--getopenfilename", os.path.expanduser("~"), "*.db *.sqlite *.sqlite3"])

    last_error = None
    for command in launchers:
        try:
            proc = subprocess.run(command, capture_output=True, text=True, timeout=300)
        except subprocess.TimeoutExpired:
            return {"error": "File dialog timed out. Type the path into the field."}
        except OSError as exc:
            last_error = str(exc)
            continue
        if proc.returncode == 0:
            chosen = (proc.stdout or "").strip()
            if not chosen:
                return {"cancelled": True}
            return {"path": os.path.abspath(chosen)}
        if proc.returncode == 1:
            return {"cancelled": True}
        last_error = (proc.stderr or "").strip() or f"dialog exited {proc.returncode}"

    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        if directory:
            chosen = filedialog.askdirectory(title=title)
        else:
            chosen = filedialog.askopenfilename(
                title=title,
                filetypes=[("SQLite", "*.db *.sqlite *.sqlite3"), ("All files", "*.*")],
            )
        root.destroy()
    except Exception as exc:
        detail = last_error or str(exc)
        return {"error": f"No file dialog is available ({detail}). Type the path into the field."}

    if not chosen:
        return {"cancelled": True}
    return {"path": os.path.abspath(chosen)}
