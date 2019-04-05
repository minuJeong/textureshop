import os
import re


def re_delaa(pattern):
    p = re.compile(pattern)

    def walk():
        excluders = [".git", ".idea", ".ropeproject", "__pycache__", ".egg-info"]
        for root, dirs, files in os.walk("./"):
            if any(filter(lambda x: x in root, excluders)):
                continue

            for filename in files:
                if not p.match(filename):
                    continue

                yield f"{root}/{filename}"

    list(map(os.remove, walk()))


# remove debug render
re_delaa(r".*[.]png$")
re_delaa(r".*[.]mp4$")

# remove temp files from rope
# starts with "tmp", 3 more letters following,
# but none of them is ".", and that's all of its name.
re_delaa(r"^tmp[^.]{6}$")
