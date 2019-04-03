
from base import Init
from noise import FBMNoise
from util import npwrite


def process():
    init = Init((512, 512))

    GL = init.get_gl()

    fbm_1 = FBMNoise(gl=GL)
    fbm_1.in_node(init)
    fbm_1_out = fbm_1.out_node()
    npwrite("fbm_1_out.png", fbm_1_out)

    import numpy as np
    noise = np.random.uniform(0.0, 1.0, (2048, 2048, 3))
    fbm_3 = FBMNoise(gl=GL)
    fbm_3.in_node(init, noise)
    fbm_3_out = fbm_3.out_node()
    npwrite("fbm_3_out.png", fbm_3_out)


if __name__ == "__main__":
    process()
