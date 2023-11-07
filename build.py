import pathlib
import sys
import subprocess
import platform


project = "j5nws"
here = pathlib.Path(__file__).resolve().parent


if platform.system() == 'Windows':
    python = 'python'
else:
    python = 'python3'


def print_with_line(s, char='#'):
    s += ' '
    s += char * (80 - len(s))
    print(s)


def print_then_call(*args, **kwargs):
    print_with_line(' '.join(args), '-')
    subprocess.run(args, check=True, **kwargs)


def fetch_data():
    print_with_line('fetch data')
    for name in ["w1w3l2l3_raw", "w1w3_passive", "w1_motorcurve", "w4_motorcurve", "w1_powercurve", "w3_powercurve"]:
        print_then_call("osf", "-p", project, "fetch", f"{name}.wt5", str(here / "data" / f"{name}.wt5"))


def build_data():
    print_with_line('workup data')
    print_then_call(python, str(here / "simulation" / "phase_matching_simulation.py"))
    print_then_call(python, str(here / "simulation" / "solve_phase_matching_angles.py"))
    print_then_call(python, str(here / "simulation" / "empirical.py"))
    print_then_call(python, str(here / "data" / "collate_data.py"))


def build_figures():
    print_with_line('figures')
    # print_then_call(python, here / "figures" / 'fig1 partial - boxcars naming conventions.py')
    print_then_call(python, str(here / "figures" / 'fig2 - 2D spectra passive.py'))
    print_then_call(python, str(here / "figures" / 'fig3 - angle dependence.py'))
    print_then_call(python, str(here / "figures" / 'fig4 - correction assessments.py'))


if __name__ == '__main__':
    if len(sys.argv) == 1:
        print('no arguments given---building everything!')
        sys.argv.append('all')
    if 'fetch' in sys.argv or 'all' in sys.argv:
        fetch_data()
    if 'data' in sys.argv or 'all' in sys.argv:
        build_data()
    if 'figures' in sys.argv or 'all' in sys.argv:
        build_figures()
    print_with_line('building done!')
