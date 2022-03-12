#lint.py
import os
import sys

from pylint import lint

THRESHOLD = 4

os.system("find -regex '.*\.\(py\)' | xargs pylint --output-format=text > .pylint/pylint.log ")
score = float(open(".pylint/pylint.log", "r").readlines()[-2].strip().split(" ")[6].split("/")[0])
if score < THRESHOLD:

    print("Linter failed: Score < threshold value")

    sys.exit(1)


sys.exit(0)
