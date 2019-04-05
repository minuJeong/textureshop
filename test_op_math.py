import math
import unittest

import moderngl as mg
import numpy as np

from op_base import Init
from op_math import Num, Add, Multiply, Divide, Clamp, OneMinus, Sin, Cos, Tan, Asin, Acos, Atan2, SinH, CosH, TanH, Power, Log_Natural, Log_2


GL = mg.create_standalone_context()
init = Init(size=(1, 1), gl=GL)


class InitializeTest(unittest.TestCase):

    def test_num(self):
        print("[+] Testing const num")

        num_a = Num(value=0.1).in_node(init).out_node()
        self.assertEqual(num_a[0, 0, 0], np.float32(0.1))

        num_b = Num().in_node(init, 10.0).out_node()
        self.assertEqual(num_b[0, 0, 0], np.float32(10.0))

        num_c = Num(value=0.5).out_node()
        self.assertEqual(num_c[0, 0, 0], np.float32(0.5))

        self.assertIsInstance(num_a, np.ndarray)
        self.assertIsInstance(num_b, np.ndarray)
        self.assertIsInstance(num_c, np.ndarray)

    def test_init(self):
        print("[+] Testing init")

        num_a = Num().in_node(init, 0.1).out_node()
        self.assertEqual(num_a.shape, (1, 1, 4))

        num_b = Num(value=0.1).in_node(init).out_node()
        self.assertEqual(num_b[0, 0, 0], np.float32(0.1))

        self.assertIsInstance(num_a, np.ndarray)
        self.assertIsInstance(num_b, np.ndarray)


class OperationTest(unittest.TestCase):

    def test_add(self):
        print("[+] Testing add node")

        num_a = Num(value=0.1).in_node(init).out_node()
        num_b = Num(value=0.2).in_node(init).out_node()

        add_a = Add().in_node(init, num_a, num_b).out_node()
        self.assertEqual(add_a.shape, (1, 1, 4))
        self.assertTrue(np.all(add_a[0, 0] == 0.3))

        add_b = Add().in_node(init, num_a, 0.3).out_node()
        self.assertEqual(add_b[0, 0, 0], np.float32(0.4))

        add_c = Add().in_node(init, 0.2, 0.6).out_node()
        self.assertEqual(add_c[0, 0, 0], np.float32(0.8))

        self.assertIsInstance(num_a, np.ndarray)
        self.assertIsInstance(num_b, np.ndarray)
        self.assertIsInstance(add_a, np.ndarray)
        self.assertIsInstance(add_b, np.ndarray)
        self.assertIsInstance(add_c, np.ndarray)

    def test_mul(self):
        print("[+] Testing mul node")

        num_a = Num(value=0.5).in_node(init).out_node()
        num_b = Num(value=10.0).in_node(init).out_node()

        mult = Multiply().in_node(init, num_a, num_b).out_node()
        self.assertEqual(mult[0, 0, 0], np.float32(5.0))

        self.assertIsInstance(num_a, np.ndarray)
        self.assertIsInstance(num_b, np.ndarray)
        self.assertIsInstance(mult, np.ndarray)

    def test_div(self):
        print("[+] Testing div node")

        div_a = Divide().in_node(init, 0.6, 0.2).out_node()
        self.assertEqual(div_a[0, 0, 0], np.float32(3.0))

        num_a = Num(value=0.5).out_node()
        num_b = Num(value=2.5).out_node()
        div_b = Divide().in_node(init, num_a, num_b).out_node()
        self.assertEqual(div_b[0, 0, 0], np.float32(0.2))

        div_c = Divide().in_node(init, 0.6, 0.2).out_node()
        self.assertEqual(div_c[0, 0, 0], np.float32(3.0))

        self.assertIsInstance(num_a, np.ndarray)
        self.assertIsInstance(num_b, np.ndarray)
        self.assertIsInstance(div_a, np.ndarray)
        self.assertIsInstance(div_b, np.ndarray)
        self.assertIsInstance(div_c, np.ndarray)

    def test_clamp(self):
        print("[+] Testing clamp node")

        clamped_a = Clamp().in_node(init, 1.2).out_node()
        self.assertEqual(clamped_a[0, 0, 0], np.float32(1.0))
        self.assertIsInstance(clamped_a, np.ndarray)

    def test_oneminus(self):
        print("[+] Testing oneminus node")

        oneminus_a = OneMinus().in_node(init, 0.25).out_node()
        self.assertEqual(oneminus_a[0, 0, 0], np.float32(0.75))
        oneminus_b = OneMinus().in_node(init, oneminus_a).out_node()
        self.assertEqual(oneminus_b[0, 0, 0], np.float32(0.25))

        self.assertIsInstance(oneminus_a, np.ndarray)
        self.assertIsInstance(oneminus_b, np.ndarray)

    def test_basic_trigonometries(self):
        print("[+] Testing basic trigonometries node")

        # accpet error less than 0.0001
        PATIENCE = 1e-4

        print("\t[+][+] Testing sin..")
        sin_a = Sin().in_node(init, 0.00).out_node()
        sin_b = Sin().in_node(init, 0.75).out_node()
        sin_c = Sin().in_node(init, 1.50).out_node()
        sin_d = Sin().in_node(init, 222.0).out_node()
        self.assertLess(abs(sin_a[0, 0, 0] - math.sin(0.00)), PATIENCE)
        self.assertLess(abs(sin_b[0, 0, 0] - math.sin(0.75)), PATIENCE)
        self.assertLess(abs(sin_c[0, 0, 0] - math.sin(1.50)), PATIENCE)
        self.assertLess(abs(sin_d[0, 0, 0] - math.sin(222.0)), PATIENCE)

        print("\t[+][+] Testing cos..")
        cos_a = Cos().in_node(init, 0.00).out_node()
        cos_b = Cos().in_node(init, 0.75).out_node()
        cos_c = Cos().in_node(init, 1.50).out_node()
        cos_d = Cos().in_node(init, 222.0).out_node()
        self.assertLess(abs(cos_a[0, 0, 0] - math.cos(0.00)), PATIENCE)
        self.assertLess(abs(cos_b[0, 0, 0] - math.cos(0.75)), PATIENCE)
        self.assertLess(abs(cos_c[0, 0, 0] - math.cos(1.50)), PATIENCE)
        self.assertLess(abs(cos_d[0, 0, 0] - math.cos(222.0)), PATIENCE)

        print("\t[+][+] Testing tan..")
        tan_a = Tan().in_node(init, 0.00).out_node()
        tan_b = Tan().in_node(init, 0.75).out_node()
        tan_c = Tan().in_node(init, 1.50).out_node()
        tan_d = Tan().in_node(init, 222.0).out_node()
        self.assertLess(abs(tan_a[0, 0, 0] - math.tan(0.00)), PATIENCE)
        self.assertLess(abs(tan_b[0, 0, 0] - math.tan(0.75)), PATIENCE)
        self.assertLess(abs(tan_c[0, 0, 0] - math.tan(1.50)), PATIENCE)
        self.assertLess(abs(tan_d[0, 0, 0] - math.tan(222.0)), PATIENCE)

    def test_arc_trigonometries(self):
        print("[+] Testing arc trigonometries node")

        # accpet error less than 0.0001
        PATIENCE = 1e-4

        print("[+][+] Testing asin..")
        asin_a = Asin().in_node(init, 1.0).out_node()
        asin_b = Asin().in_node(init, 0.1).out_node()
        asin_c = Asin().in_node(init, -1.0).out_node()
        asin_d = Asin().in_node(init, -0.5).out_node()

        self.assertLess(abs(asin_a[0, 0, 0] - math.asin(1.0)), PATIENCE)
        self.assertLess(abs(asin_b[0, 0, 0] - math.asin(0.1)), PATIENCE)
        self.assertLess(abs(asin_c[0, 0, 0] - math.asin(-1.0)), PATIENCE)
        self.assertLess(abs(asin_d[0, 0, 0] - math.asin(-0.5)), PATIENCE)

        print("\t[+][+] Testing acos..")
        acos_a = Acos().in_node(init, 1.0).out_node()
        acos_b = Acos().in_node(init, 0.1).out_node()
        acos_c = Acos().in_node(init, -1.0).out_node()
        acos_d = Acos().in_node(init, -0.5).out_node()

        self.assertLess(abs(acos_a[0, 0, 0] - math.acos(1.0)), PATIENCE)
        self.assertLess(abs(acos_b[0, 0, 0] - math.acos(0.1)), PATIENCE)
        self.assertLess(abs(acos_c[0, 0, 0] - math.acos(-1.0)), PATIENCE)
        self.assertLess(abs(acos_d[0, 0, 0] - math.acos(-0.5)), PATIENCE)

        print("\t[+][+] Testing atan2..")
        atan2_a = Atan2().in_node(init, 1.0, 2.0).out_node()
        atan2_b = Atan2().in_node(init, 0.1, 23.2).out_node()
        atan2_c = Atan2().in_node(init, 10.5, 0.11).out_node()
        atan2_d = Atan2().in_node(init, -100.0, 0.0).out_node()

        self.assertLess(abs(atan2_a[0, 0, 0] - math.atan2(1.0, 2.0)), PATIENCE)
        self.assertLess(abs(atan2_b[0, 0, 0] - math.atan2(0.1, 23.2)), PATIENCE)
        self.assertLess(abs(atan2_c[0, 0, 0] - math.atan2(10.5, 0.11)), PATIENCE)
        self.assertLess(abs(atan2_d[0, 0, 0] - math.atan2(-100.0, 0.0)), PATIENCE)

    def test_hyperbolic_trigonometries(self):
        print("[+] Testing hyperbolic trigonometries node")

        # accpet error less than 0.0001
        PATIENCE = 1e-4

        print("\t[+][+] Testing sinh..")
        sinh_a = SinH().in_node(init, 1.0).out_node()
        sinh_b = SinH().in_node(init, 0.1).out_node()
        sinh_c = SinH().in_node(init, 2.0).out_node()

        self.assertLess(abs(sinh_a[0, 0, 0] - math.sinh(1.0)), PATIENCE)
        self.assertLess(abs(sinh_b[0, 0, 0] - math.sinh(0.1)), PATIENCE)
        self.assertLess(abs(sinh_c[0, 0, 0] - math.sinh(2.0)), PATIENCE)

        print("\t[+][+] Testing cosh..")
        cosh_a = CosH().in_node(init, 1.0).out_node()
        cosh_b = CosH().in_node(init, 0.1).out_node()
        cosh_c = CosH().in_node(init, 2.0).out_node()

        self.assertLess(abs(cosh_a[0, 0, 0] - math.cosh(1.0)), PATIENCE)
        self.assertLess(abs(cosh_b[0, 0, 0] - math.cosh(0.1)), PATIENCE)
        self.assertLess(abs(cosh_c[0, 0, 0] - math.cosh(2.0)), PATIENCE)

        print("\t[+][+] Testing tanh..")
        tanh_a = TanH().in_node(init, 1.0).out_node()
        tanh_b = TanH().in_node(init, 0.1).out_node()
        tanh_c = TanH().in_node(init, 2.0).out_node()

        self.assertLess(abs(tanh_a[0, 0, 0] - math.tanh(1.0)), PATIENCE)
        self.assertLess(abs(tanh_b[0, 0, 0] - math.tanh(0.1)), PATIENCE)
        self.assertLess(abs(tanh_c[0, 0, 0] - math.tanh(2.0)), PATIENCE)

    def test_power_log(self):
        print("[+] Testing power/log node")

        pow_a = Power().in_node(init, 2.0, 2.0).out_node()
        pow_b = Power().in_node(init, 1.1, 10.0).out_node()
        pow_c = Power().in_node(init, 2.0, 10.0).out_node()
        pow_d = Power().in_node(init, 5.3, 2.1).out_node()

        self.assertTrue(np.all(np.isclose(pow_a, np.power(2.0, 2.0))))
        self.assertTrue(np.all(np.isclose(pow_b, np.power(1.1, 10.0))))
        self.assertTrue(np.all(np.isclose(pow_c, np.power(2.0, 10.0))))
        self.assertTrue(np.all(np.isclose(pow_d, np.power(5.3, 2.1))))

        log_a = Log_Natural().in_node(init, 12.53).out_node()
        log_b = Log_Natural().in_node(init, 22.23).out_node()

        self.assertTrue(np.all(np.isclose(log_a, np.log(12.53))))
        self.assertTrue(np.all(np.isclose(log_b, np.log(22.23))))

        log2_a = Log_2().in_node(init, 12.53).out_node()
        log2_b = Log_2().in_node(init, 22.23).out_node()

        self.assertTrue(np.all(np.isclose(log2_a, np.log2(12.53))))
        self.assertTrue(np.all(np.isclose(log2_b, np.log2(22.23))))


if __name__ == "__main__":
    unittest.main()
