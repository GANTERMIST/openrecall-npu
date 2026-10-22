"""Логика лаунчера WinAI.exe: когда нужна установка и что запускать."""
import socket

import launcher


def test_needs_setup_tracks_requirements(tmp_path, monkeypatch):
    (tmp_path / "requirements.txt").write_text("numpy\n")
    venv = tmp_path / ".venv"
    monkeypatch.setattr(launcher, "APP", tmp_path)
    monkeypatch.setattr(launcher, "VENV", venv)
    monkeypatch.setattr(launcher, "STAMP", venv / ".winai-requirements")
    assert launcher.needs_setup()  # окружения нет
    py = launcher.venv_python()
    py.parent.mkdir(parents=True)
    py.write_text("")
    assert launcher.needs_setup()  # окружение есть, но не помечено как готовое
    launcher.STAMP.write_text(launcher.requirements_hash())
    assert not launcher.needs_setup()
    (tmp_path / "requirements.txt").write_text("numpy\nflask\n")
    assert launcher.needs_setup()  # обновление добавило зависимость


def test_running_instance_just_opens_panel(monkeypatch):
    srv = socket.socket()
    srv.bind(("127.0.0.1", 0))
    srv.listen()
    monkeypatch.setattr(launcher, "PORT", srv.getsockname()[1])
    opened, launched = [], []
    monkeypatch.setattr(launcher.webbrowser, "open", opened.append)
    monkeypatch.setattr(launcher, "launch", lambda: launched.append(1))
    assert launcher.main([]) == 0
    assert opened and not launched
    srv.close()
