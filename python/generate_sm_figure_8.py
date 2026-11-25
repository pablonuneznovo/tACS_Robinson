"""Generate Supplementary Figure S8 using the same plot code as Figure 2."""
import runpy
import sys

if __name__ == '__main__':
    sys.argv[1:] = ['--cohort', 'stbi']
    runpy.run_module('generate_figure_2', run_name='__main__')
