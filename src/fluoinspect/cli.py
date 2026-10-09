"""Command line entry points with lazy loading of image-processing modules."""
import sys


def main():
    commands = {"measure": "pipeline", "inspect": "investigation.session",
                "packet": "tools.evidence", "deploy": "deployment.plan"}
    if len(sys.argv) == 1 or sys.argv[1] in {"-h", "--help"}:
        print("FluoInspect: fluorescence and autofluorescence QC\n"
              "Usage: fluoinspect {measure,inspect,packet,deploy} [options]\n"
              "  measure  Native intensity, detail, background and axial-pattern evidence\n"
              "  inspect  Create sessions, request views, audit or validate reports\n"
              "  packet   Prepare verified views and source-bound measurements\n"
              "  deploy   Plan node assignments or check local runtime configuration")
        return
    command = sys.argv[1]
    if command == "--version":
        from . import __version__
        print(__version__)
        return
    if command not in commands:
        raise SystemExit(f"Unknown command: {command}")
    import importlib
    module = importlib.import_module("fluoinspect." + commands[command])
    sys.argv = [sys.argv[0], *sys.argv[2:]]
    module.main()
