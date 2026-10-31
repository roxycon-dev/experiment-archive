"""Current table-generation entry; previous generation preserved in revisions/."""
from pathlib import Path
import runpy
runpy.run_path(str(Path(__file__).with_name('generate_supplement_tables.py')),run_name='__main__')
