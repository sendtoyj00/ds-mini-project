"""Run the battery lifetime experiment."""
import runpy
from src.train import main

if __name__ == "__main__":
    runpy.run_module("src.train", run_name="__main__")
