import unittest
import feng

class Tests(unittest.TestCase):

    def test_submodules_exported(self):
        submodules = ('importance', 'pipeline', 'preprocessing')
        members = dir(feng)
        self.assertTrue(all(module in members for module in submodules))


if __name__ == '__main__':
    all_tests = unittest.TestLoader().loadTestsFromTestCase(Tests)
    unittest.TextTestRunner(verbosity=2).run(all_tests)
