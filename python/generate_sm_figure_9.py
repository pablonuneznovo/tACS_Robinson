"""Generate Supplementary Figure S9: the acute within-session changes."""
import runpy
import sys

if __name__ == '__main__':
    sys.argv[1:] = ['--acute-only']
    runpy.run_module('generate_figure_2', run_name='__main__')
