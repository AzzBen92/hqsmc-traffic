"""Low-level helpers to locate and start SUMO through TraCI (or libsumo)."""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path


def find_sumo_home(configured: str = "") -> Path:
    """Return SUMO_HOME: config value, then environment variable, then the pip package `eclipse-sumo`."""
    for candidate in (configured, os.environ.get("SUMO_HOME", "")):
        if candidate and Path(candidate).exists():
            return Path(candidate)
    try:  # pip install eclipse-sumo
        import sumo  # type: ignore
        return Path(sumo.SUMO_HOME)
    except ImportError:
        pass
    exe = shutil.which("sumo")
    if exe:
        return Path(exe).resolve().parent.parent
    raise RuntimeError(
        "SUMO not found. Install SUMO 1.27+ (https://sumo.dlr.de) and set the SUMO_HOME "
        "environment variable, or run `pip install eclipse-sumo traci sumolib`."
    )


def add_sumo_tools_to_path(sumo_home: Path) -> None:
    tools = sumo_home / "tools"
    if tools.exists() and str(tools) not in sys.path:
        sys.path.append(str(tools))


def sumo_binary(sumo_home: Path, gui: bool = False) -> str:
    name = "sumo-gui" if gui else "sumo"
    for path in (sumo_home / "bin" / name, sumo_home / "bin" / f"{name}.exe"):
        if path.exists():
            return str(path)
    exe = shutil.which(name)
    if exe:
        return exe
    raise RuntimeError(f"Cannot find the '{name}' executable under {sumo_home / 'bin'}.")


def get_traci(use_libsumo: bool = False):
    """Return the TraCI-compatible module (libsumo is faster but has no GUI)."""
    if use_libsumo:
        import libsumo as traci  # type: ignore
    else:
        import traci  # type: ignore
    return traci


class SumoRunner:
    """Start / stop a SUMO instance for one episode."""

    def __init__(self, cfg, label: str = "default"):
        self.cfg = cfg
        self.label = label
        self.sumo_home = find_sumo_home(cfg.SUMO_HOME)
        add_sumo_tools_to_path(self.sumo_home)
        self.use_libsumo = bool(cfg.USE_LIBSUMO) and not cfg.SUMO_GUI
        self.traci = get_traci(self.use_libsumo)
        self.running = False
        if not Path(cfg.SUMO_CFG).exists():
            raise FileNotFoundError(
                f"SUMO configuration not found: {cfg.SUMO_CFG}\n"
                "Put your network/demand files in map/<scenario>/ (see map/README.md) or build "
                "the synthetic test networks with `python scripts/make_test_networks.py`."
            )

    def command(self, seed: int) -> list[str]:
        cfg = self.cfg
        cmd = [
            sumo_binary(self.sumo_home, cfg.SUMO_GUI),
            "-c", str(cfg.SUMO_CFG),
            "--seed", str(seed),
            "--step-length", str(cfg.STEP_LENGTH),
            # one extra step: the initial observation uses one step, then T decisions follow
            "--end", str(cfg.EPISODE_LENGTH + cfg.STEP_LENGTH),
            "--time-to-teleport", str(cfg.TIME_TO_TELEPORT),
            "--waiting-time-memory", str(int(cfg.EPISODE_LENGTH)),
            "--no-step-log", "true",
            "--no-warnings", "true",
            "--duration-log.disable", "true",
        ]
        if cfg.SUMO_GUI:
            cmd += ["--start", "true"]
        cmd += list(cfg.SUMO_EXTRA_ARGS)
        return cmd

    def start(self, seed: int):
        """Start SUMO for a new episode.

        The process is launched once; the following episodes reload the simulation with
        traci.load(), which is much faster than restarting SUMO (and gives the same result).
        """
        cmd = self.command(seed)
        if self.running:
            try:
                if not self.use_libsumo:
                    self.traci.switch(self.label)
                self.traci.load(cmd[1:])
                return self.traci
            except Exception:  # noqa: BLE001 - fall back to a full restart
                self.close()
        if self.use_libsumo:
            self.traci.start(cmd)
        else:
            self.traci.start(cmd, label=self.label)
            self.traci.switch(self.label)
        self.running = True
        return self.traci

    def close(self):
        if self.running:
            try:
                if not self.use_libsumo:
                    self.traci.switch(self.label)
                self.traci.close()
            except Exception:  # noqa: BLE001 - SUMO may already be closed
                pass
            self.running = False
