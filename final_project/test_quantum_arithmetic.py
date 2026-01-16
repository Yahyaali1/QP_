"""
Unit tests for quantum arithmetic operations
"""

import unittest
from qiskit import QuantumCircuit, transpile
from qiskit_aer import Aer

from final_project.quantum_arithmetic import (
    set_bits, copy, full_adder, add, subtract,
    greater_or_eq, add_mod, times_two_mod,
    times_two_power_mod, multiply_mod
)


class TestQuantumArithmetic(unittest.TestCase):

    def get_counts(self, qc):
        """Helper to run circuit and get counts"""
        backend = Aer.get_backend('qasm_simulator')
        result = backend.run(transpile(qc, backend), shots=1024).result()
        return result.get_counts()

    def test_initialization(self):
        """Test set_bits function"""
        qubits_A = [2, 4, 3, 7, 5]
        input_X = "01011"

        n_qubits = 8
        qc = QuantumCircuit(n_qubits, n_qubits)
        set_bits(qc, qubits_A, input_X)

        qc.measure(qubits_A, range(len(qubits_A)))

        simulator = Aer.get_backend('qasm_simulator')
        result = simulator.run(transpile(qc, simulator), shots=1).result()
        counts = result.get_counts()

        measured_state = list(counts.keys())[0]
        print(f"Test Initialization: {measured_state[::-1]}")

    def test_copy_function(self):
        """Test copy function"""
        reg_A = [0, 1, 2]
        reg_B = [3, 4, 5]
        input_pattern = "110"

        qc = QuantumCircuit(6, 3)

        # Initialize A
        set_bits(qc, reg_A, input_pattern)

        # Copy
        copy(qc, reg_A, reg_B)

        # Measure B
        qc.measure(reg_B, [0, 1, 2])

        simulator = Aer.get_backend('qasm_simulator')
        result = simulator.run(transpile(qc, simulator), shots=1).result()
        counts = result.get_counts()
        measured_state = list(counts.keys())[0]

        print(f"Test Copy: Expected {input_pattern}, Got {measured_state[::-1]}")

    def test_full_adder(self):
        """Test full adder"""
        truth_table = {
            (0, 1, 0): (1, 0),
        }

        for (a, b, cin), (exp_sum, exp_cout) in truth_table.items():
            qc = QuantumCircuit(6, 2)

            if a:
                set_bits(qc, [0], "1")
            if b:
                set_bits(qc, [1], "1")
            if cin:
                set_bits(qc, [2], "1")

            # Apply Full Adder
            full_adder(qc, 0, 1, 3, 2, 4, [5])

            qc.measure(3, 0)  # Read r
            qc.measure(4, 1)  # Read cout

            counts = self.get_counts(qc)
            print(f"Test Full Adder: {a} + {b} + {cin} -> {counts}")

    def test_add(self):
        """Test 4-bit Ripple Carry Addition: 7 + 5 = 12"""
        n = 4
        qa = list(range(0, 4))
        qb = list(range(4, 8))
        qr = list(range(8, 12))
        aux = list(range(12, 18))

        qc = QuantumCircuit(18, 4)

        # Initialize: 7 (0111) + 5 (0101)
        set_bits(qc, qa, "1110")
        set_bits(qc, qb, "1010")

        # Execute
        add(qc, qa, qb, qr, aux)

        # Measure Result R
        qc.measure(qr, range(4))

        # Verify
        counts = self.get_counts(qc)
        measured_bin = max(counts, key=counts.get)
        measured_int = int(measured_bin, 2)

        print(f"Test Add: 7 + 5 = {measured_int} (Binary: {measured_bin})")
        self.assertEqual(measured_int, 12)

    def test_subtract(self):
        """Test 4-bit Subtraction: 10 - 3 = 7"""
        n = 4
        qa = list(range(0, 4))
        qb = list(range(4, 8))
        qr = list(range(8, 12))
        aux = list(range(12, 18))

        qc = QuantumCircuit(18, 4)

        # Initialize: 10 (1010) - 3 (0011)
        set_bits(qc, qa, "0101")
        set_bits(qc, qb, "1100")

        # Execute
        subtract(qc, qa, qb, qr, aux)

        # Measure Result R
        qc.measure(qr, range(4))

        # Verify
        counts = self.get_counts(qc)
        measured_bin = max(counts, key=counts.get)
        measured_int = int(measured_bin, 2)

        print(f"Test Sub: 10 - 3 = {measured_int} (Binary: {measured_bin})")
        self.assertEqual(measured_int, 7)

    def test_greater_eq(self):
        """Test Greater Than or Equal: A >= B"""
        test_cases = [
            (5, "1010", 3, "1100", 1),  # 5 >= 3 (True)
            (2, "0100", 5, "1010", 0)  # 2 >= 5 (False)
        ]

        for a_int, a_str, b_int, b_str, expected in test_cases:
            with self.subTest(f"{a_int} >= {b_int}"):
                n = 4
                qa = list(range(0, 4))
                qb = list(range(4, 8))
                qr = 8
                aux = list(range(9, 14))

                qc = QuantumCircuit(14, 1)

                # Initialize A and B
                set_bits(qc, qa, a_str)
                set_bits(qc, qb, b_str)

                # Execute
                greater_or_eq(qc, qa, qb, qr, aux)

                # Measure Result
                qc.measure(qr, 0)

                # Verify
                counts = self.get_counts(qc)
                measured_bin = max(counts, key=counts.get)
                measured_int = int(measured_bin, 2)

                print(f"Test >= : {a_int} >= {b_int} -> Result: {measured_int}")
                self.assertEqual(measured_int, expected)

    def test_add_mod(self):
        """Test Modular Addition: R = (A + B) % N"""
        test_cases = [
            (11, 2, 3, 5),  # 2+3=5 (<11)
            (5, 3, 4, 2),  # 3+4=7 (>=5). 7-5=2
            (5, 2, 3, 0)  # 2+3=5 (>=5). 5-5=0
        ]

        for n_val, a_val, b_val, expected in test_cases:
            with self.subTest(f"{a_val} + {b_val} mod {n_val}"):
                n = 4
                qn = list(range(0, 4))
                qa = list(range(4, 8))
                qb = list(range(8, 12))
                qr = list(range(12, 16))
                aux = list(range(16, 23))

                qc = QuantumCircuit(23, 4)

                def set_val(reg, val):
                    bin_str = format(val, f'0{n}b')[::-1]
                    for i, bit in enumerate(bin_str):
                        if bit == '1':
                            qc.x(reg[i])

                set_val(qn, n_val)
                set_val(qa, a_val)
                set_val(qb, b_val)

                # Execute
                add_mod(qc, qn, qa, qb, qr, aux)

                # Verify
                qc.measure(qr, range(4))
                counts = self.get_counts(qc)
                measured_bin = max(counts, key=counts.get)
                measured_int = int(measured_bin, 2)

                print(f"({a_val} + {b_val}) % {n_val} = {measured_int}")
                self.assertEqual(measured_int, expected)

    def test_times_two_mod(self):
        """Test Multiplication by Two Modulo N"""
        test_cases = [
            (10, 2, 4),  # 2*2 = 4 (mod 10)
            (5, 3, 1),  # 2*3 = 6 -> 1 (mod 5)
            (4, 2, 0)  # 2*2 = 4 -> 0 (mod 4)
        ]

        print("\n--- Testing Times Two Modulo ---")

        for n_val, a_val, expected in test_cases:
            with self.subTest(f"2 * {a_val} mod {n_val}"):
                n = 4
                qn = list(range(0, 4))
                qa = list(range(4, 8))
                qr = list(range(8, 12))
                aux = list(range(12, 26))

                qc = QuantumCircuit(26, 4)

                # Initialize
                set_bits(qc, qn, format(n_val, f'0{n}b')[::-1])
                set_bits(qc, qa, format(a_val, f'0{n}b')[::-1])

                # Execute
                times_two_mod(qc, qn, qa, qr, aux)

                # Verify
                qc.measure(qr, range(4))
                counts = self.get_counts(qc)
                measured_int = int(max(counts, key=counts.get), 2)

                print(f"(2 * {a_val}) % {n_val} = {measured_int}")
                self.assertEqual(measured_int, expected)

    def test_times_two_power_mod(self):
        """Test Multiplication by 2^k Modulo N"""
        test_cases = [
            # (15, 3, 1, 6),    # 2^1 * 3 = 6 mod 15
            # (15, 3, 2, 12),   # 2^2 * 3 = 12 mod 15
            (15, 3, 3, 9),    # 2^3 * 3 = 24 = 9 mod 15
            # (7, 4, 2, 2),  # 2^2 * 4 = 16 = 2 mod 7
            # (5, 3, 3, 4)      # 2^3 * 3 = 24 = 4 mod 5
        ]

        print("\n--- Testing Times Two Power Modulo ---")

        for n_val, a_val, k, expected in test_cases:
            with self.subTest(f"2^{k} * {a_val} mod {n_val}"):
                n = 4
                qn = list(range(0, 4))
                qa = list(range(4, 8))
                qr = list(range(8, 12))
                aux = list(range(12, 12 + (3 * n + 2)))

                qc = QuantumCircuit(12 + (3 * n + 2), 4)

                # Initialize N and A
                set_bits(qc, qn, format(n_val, f'0{n}b')[::-1])
                set_bits(qc, qa, format(a_val, f'0{n}b')[::-1])

                # Execute
                times_two_power_mod(qc, qn, qa, k, qr, aux)

                # Measure result
                qc.measure(qr, range(4))
                counts = self.get_counts(qc)
                measured_bin = max(counts, key=counts.get)
                measured_int = int(measured_bin, 2)

                print(f"(2^{k} * {a_val}) % {n_val} = {measured_int}")
                self.assertEqual(measured_int, expected)

    def test_multiply_mod(self):
        """Test Modular Multiplication: R = (A * B) % N"""
        test_cases = [
            (15, 3, 5, 0),  # 3*5 = 15 -> 0 mod 15
            (7, 3, 4, 5),  # 3*4 = 12 -> 5 mod 7
            (11, 7, 8, 1),  # 7*8 = 56 -> 1 mod 11
            (13, 6, 6, 10)  # 6*6 = 36 -> 10 mod 13
        ]

        print("\n--- Testing Multiply Modulo ---")

        for n_val, a_val, b_val, expected in test_cases:
            with self.subTest(f"{a_val} * {b_val} mod {n_val}"):
                n = 4
                qn = list(range(0, 4))
                qa = list(range(4, 8))
                qb = list(range(8, 12))
                qr = list(range(12, 16))
                aux = list(range(16, 16 + (4 * n + 2)))

                qc = QuantumCircuit(16 + (4 * n + 2), 4)

                # Initialize registers
                set_bits(qc, qn, format(n_val, f'0{n}b')[::-1])
                set_bits(qc, qa, format(a_val, f'0{n}b')[::-1])
                set_bits(qc, qb, format(b_val, f'0{n}b')[::-1])

                # Execute multiplication
                multiply_mod(qc, qn, qa, qb, qr, aux)

                # Measure result
                qc.measure(qr, range(4))
                counts = self.get_counts(qc)
                measured_bin = max(counts, key=counts.get)
                measured_int = int(measured_bin, 2)

                print(f"({a_val} * {b_val}) % {n_val} = {measured_int}")
                self.assertEqual(measured_int, expected)


if __name__ == '__main__':
    unittest.main()