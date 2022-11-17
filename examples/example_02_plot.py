"""Plot the results of Example 1."""

import matplotlib.pyplot as plt
import numpy as np
from argparse import ArgumentParser
import re

parser = ArgumentParser()
parser.add_argument(
  "input_files",
  metavar="FILE",
  type=str,
  nargs="+",
  help="the filenames to process"
)
parser.add_argument(
    "--skip",
    dest="skip",
    help="skip this many samples",
    type=int,
    default=0
)

args = parser.parse_args()

r = re.compile(r"gamma=(\d\.\d+)_s=(\d\.\d+e-\d+)_n=(\d+)")

fig, ax = plt.subplots()

for f in args.input_files:
    m = r.search(f)
    gamma = float(m.group(1))
    s = float(m.group(2))
    n = int(m.group(3))
    data = np.loadtxt(f)
    ax.boxplot(data[args.skip:], positions=[gamma])
#ax.set_yscale("log")
plt.show()
