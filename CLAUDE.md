WITCH
=============
The aim of this project is to implement needlet based multi resolution mapmaking. 
General
-------
- For any new modules, you can initialize a git repository and commit as needed to track changes
- The virtual env ~/venvs/general already exists to support this software package but feel free to add more packages.
- Install and use uv to manage Python packages
- You may suggest improvements or fixes to the software in .utils
- Write reports to html that include figures from test plots. Always update the relevant report as the project progresses.
- 
Coding guidelines
-----------------
- Reduce code duplication by reusing repeated parts, and by using utility packages from .utils
- Write tight, concise, readable code
- Avoid type hinting and other distractions
- Write full docstrings for each function
- Any scripts you write (for testing or otherwise) should have argparse CLI and a --debug option for quick tests (e.g. with less number of simulations or objects and/or lower resolution)
- Run ruff linting and formatting always before running code
