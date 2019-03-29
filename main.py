
from base import Init
from noise import FBMNoise
from util import npwrite


def process():
    init = Init()

    GL = init.get_gl()

    fbm = FBMNoise(GL)
    fbm.in_node(init)
    fbm_out = fbm.out_node()

    npwrite("output.png", fbm_out)


if __name__ == "__main__":
    process()
