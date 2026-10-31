"""Set only CPU threading, then run the archived MMSeg training entry point."""
from pathlib import Path
import runpy
import sys
import torch
import cv2
torch.set_num_threads(1);cv2.setNumThreads(1)
repo=Path(__file__).resolve().parent/'repo'
sys.path.insert(0,str(repo))
runpy.run_path(str(repo/'tools/train.py'),run_name='__main__')
