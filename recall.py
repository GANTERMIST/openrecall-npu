"""Совместимость со старыми командами: python recall.py capture|search|purge|serve."""
import sys

from winai import panel, recall

if __name__ == "__main__":
    if sys.argv[1:2] == ["serve"]:
        panel.run()
    else:
        recall.main(sys.argv[1:])
