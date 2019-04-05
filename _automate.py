import os
import re


def _remove_temp(pattern):
    p = re.compile(pattern)
    files = os.listdir("./")
    list(map(os.remove, filter(p.match, files)))


_remove_temp(".*[.]png$")
_remove_temp(".*[.]mp4$")
