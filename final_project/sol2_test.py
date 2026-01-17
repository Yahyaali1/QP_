"""
Unit tests for quantum arithmetic operations
"""

import unittest
from qiskit import QuantumCircuit, transpile
from qiskit.transpiler import CouplingMap
from qiskit_aer import AerSimulator

# Importing functions. Assumes final_project package structure exists.
# If running locally without package structure, ensure imports match your file layout.
from final_project.sol2 import (
    set_bits, copy, full_adder, add, subtract,
    greater_or_eq, add_mod, times_two_mod,
    times_two_power_mod, multiply_mod,
    multiply_mod_fixed, multiply_mod_fixed_power_2_k,
    multiply_mod_fixed_power_Y
)
import final_project.quantum_arithmetic as qa_module


class TestQuantumArithmetic(unittest.TestCase):
    def get_counts_intense(self, qc):
        simulator = AerSimulator(method="matrix_product_state")
        nq = qc.num_qubits
        coupling_map = CouplingMap.from_full(nq)

        transpiled_circuit = transpile(
            qc,
            basis_gates=["u", "cx"],
            coupling_map=coupling_map,
            optimization_level=0
        )

        result = simulator.run(transpiled_circuit, shots=1024).result()
        return result.get_counts()

    def get_counts(self, qc):
        """
        Helper to run circuit and get counts.
        Uses matrix_product_state to handle larger qubit counts required by
        modular multiplication functions.
        """
        backend = AerSimulator(method='matrix_product_state')
        # Optimization level 0 to preserve circuit structure during debugging
        result = backend.run(transpile(qc, backend, optimization_level=0), shots=1024).result()
        return result.get_counts()

    # 1. Test set_bits
    def test_set_bits(self):
        """Test initialization of bits"""
        n_qubits = 4
        qc = QuantumCircuit(n_qubits, n_qubits)
        target_qubits = [0, 1, 2, 3]

        # Set 10 (binary 1010). Note: Implementation usually treats index 0 as LSB.
        # If set_bits reverses input, "1010" -> LSB=1, MSB=0.
        # Using 6 (0110) which is palindrome-ish to be safe, or 5 (0101).
        val = 5  # Binary 0101
        set_bits(qc, target_qubits, val)

        qc.measure(target_qubits, range(n_qubits))
        counts = self.get_counts(qc)
        measured = int(list(counts.keys())[0], 2)
        self.assertEqual(measured, val)

    # 2. Test copy
    def test_copy(self):
        """Test copy register"""
        n = 3
        qc = QuantumCircuit(2 * n, n)
        A = [0, 1, 2]
        B = [3, 4, 5]

        val = 6  # 110
        set_bits(qc, A, val)
        copy(qc, A, B)

        qc.measure(B, range(n))
        counts = self.get_counts(qc)
        measured = int(list(counts.keys())[0], 2)
        self.assertEqual(measured, val)

    # 3. Test full_adder
    def test_full_adder(self):
        """Test single bit full adder"""
        # Case: 1 + 1 + 0 (carry_in) = 0 (sum), 1 (carry_out)
        qc = QuantumCircuit(6, 2)
        a, b, cin = 0, 1, 2
        r, cout = 3, 4
        aux = [5]

        # Set a=1, b=1
        qc.x(a)
        qc.x(b)

        full_adder(qc, a, b, r, cin, cout, aux)

        qc.measure(r, 0)
        qc.measure(cout, 1)

        counts = self.get_counts(qc)
        # Expected '10' -> cout=1, r=0 (qiskit reads right-to-left)
        self.assertIn('10', counts)

    # 4. Test add
    def test_add(self):
        """Test integer addition"""
        n = 4
        qc = QuantumCircuit(3 * n + 2, n)  # A, B, R, AUX

        A = list(range(0, n))
        B = list(range(n, 2 * n))
        R = list(range(2 * n, 3 * n))
        AUX = list(range(3 * n, 3 * n + n + 1))  # Need n+1 aux

        # 3 + 6 = 9
        set_bits(qc, A, 3)
        set_bits(qc, B, 6)

        add(qc, A, B, R, AUX)

        qc.measure(R, range(n))
        counts = self.get_counts(qc)
        measured = int(list(counts.keys())[0], 2)
        self.assertEqual(measured, 9)

    # 5. Test subtract
    def test_subtract(self):
        """Test integer subtraction"""
        n = 4
        qc = QuantumCircuit(3 * n + 2, n)

        A = list(range(0, n))
        B = list(range(n, 2 * n))
        R = list(range(2 * n, 3 * n))
        AUX = list(range(3 * n, 3 * n + n + 1))

        # 12 - 5 = 7
        set_bits(qc, A, 12)
        set_bits(qc, B, 5)

        subtract(qc, A, B, R, AUX)

        qc.measure(R, range(n))
        counts = self.get_counts(qc)
        measured = int(list(counts.keys())[0], 2)
        self.assertEqual(measured, 7)

    # 6. Test greater_or_eq
    def test_greater_or_eq(self):
        """Test comparison"""
        n = 4
        # A(4), B(4), r(1), AUX(n+1)
        qc = QuantumCircuit(2 * n + 1 + n + 1, 1)

        A = list(range(0, n))
        B = list(range(n, 2 * n))
        r = 2 * n
        AUX = list(range(2 * n + 1, 3 * n + 2))

        # 7 >= 5 -> True (1)
        set_bits(qc, A, 7)
        set_bits(qc, B, 5)

        greater_or_eq(qc, A, B, r, AUX)

        qc.measure(r, 0)
        counts = self.get_counts(qc)
        # Expect '1'
        self.assertIn('1', counts)

    # 7. Test add_mod
    def test_add_mod(self):
        """Test modular addition"""
        n = 4
        aux_req = 2 * n + 2
        total_qubits = 4 * n + aux_req
        qc = QuantumCircuit(total_qubits, n)

        A = list(range(0, n))
        B = list(range(n, 2 * n))
        N = list(range(2 * n, 3 * n))
        R = list(range(3 * n, 4 * n))
        AUX = list(range(4 * n, 4 * n + aux_req))

        # 9 + 5 mod 11 = 14 mod 11 = 3
        set_bits(qc, N, 11)
        set_bits(qc, A, 9)
        set_bits(qc, B, 5)

        add_mod(qc, N, A, B, R, AUX)

        qc.measure(R, range(n))
        counts = self.get_counts(qc)
        measured = int(list(counts.keys())[0], 2)
        self.assertEqual(measured, 3)

    # 8. Test times_two_mod
    def test_times_two_mod(self):
        """Test doubling modulo N"""
        n = 4
        aux_req = 3 * n + 2
        total_qubits = 3 * n + aux_req
        qc = QuantumCircuit(total_qubits, n)

        A = list(range(0, n))
        N = list(range(n, 2 * n))
        R = list(range(2 * n, 3 * n))
        AUX = list(range(3 * n, 3 * n + aux_req))

        # 2 * 4 mod 6 = 8 mod 6 = 2
        set_bits(qc, N, 6)
        set_bits(qc, A, 4)

        times_two_mod(qc, N, A, R, AUX)

        qc.measure(R, range(n))
        counts = self.get_counts(qc)
        measured = int(list(counts.keys())[0], 2)
        self.assertEqual(measured, 2)

    # 9. Test times_two_power_mod
    def test_times_two_power_mod(self):
        """Test multiplication by power of 2 modulo N"""
        n = 4  # Reduced n to save simulation time
        k = 2

        # Logic from proj2: aux_len = (k + 1) * n + (3*n + 2)
        aux_len = (k + 1) * n + (3 * n + 2)
        total_qubits = 3 * n + aux_len
        qc = QuantumCircuit(total_qubits, n)

        N = list(range(0, n))
        A = list(range(n, 2 * n))
        R = list(range(2 * n, 3 * n))
        AUX = list(range(3 * n, 3 * n + aux_len))

        # N=5, A=2, k=2.
        # 2 * (2^2) = 8. 8 mod 5 = 3.

        # N= 7, A= 3 , K= 2   = 5
        #
        set_bits(qc, N, 7)
        set_bits(qc, A, 3)

        times_two_power_mod(qc, N, A, k, R, AUX)

        qc.measure(R, range(n))
        counts = self.get_counts(qc)
        measured = int(list(counts.keys())[0], 2)
        self.assertEqual(measured, 5)

    # 10. Test multiply_mod
    def test_multiply_mod(self):
        """Test general modular multiplication"""
        n = 4  # Reduce n to manage high qubit count

        # Logic from proj2: aux_len = 2*n*n + 8*n + 4
        aux_len = 2 * n * n + 8 * n + 4
        total_qubits = 4 * n + aux_len
        qc = QuantumCircuit(total_qubits, n)

        N = list(range(0, n))
        A = list(range(n, 2 * n))
        B = list(range(2 * n, 3 * n))
        R = list(range(3 * n, 4 * n))
        AUX = list(range(4 * n, 4 * n + aux_len))

        # N=7, A=3, B=4.
        # 3 * 4 = 12. 12 mod 7 = 5.
        set_bits(qc, N, 7)
        set_bits(qc, A, 3)
        set_bits(qc, B, 4)

        multiply_mod(qc, N, A, B, R, AUX)

        qc.measure(R, range(n))
        counts = self.get_counts_intense(qc)
        measured = int(list(counts.keys())[0], 2)
        self.assertEqual(measured, 5)

    # 11. Test multiply_mod_fixed
    def test_multiply_mod_fixed(self):
        """Test multiplication by fixed constant X"""
        n = 4
        N_val = 7
        X = 2

        # Required to inject N_val for the function's classical pre-computation
        # (The proj2 code expects N_val in global scope)
        qa_module.N_val = N_val

        # Aux calculation based on multiply_mod requirements
        aux_mul_len = 2 * n * n + 8 * n + 4
        aux_len = 3 * n + aux_mul_len
        total_qubits = 2 * n + aux_len
        qc = QuantumCircuit(total_qubits, n)

        N = list(range(0, n))
        B = list(range(n, 2 * n))
        AUX = list(range(2 * n, 2 * n + aux_len))

        # N=7, B=3, X=2 (fixed).
        # 2 * 3 = 6. 6 mod 7 = 6.
        set_bits(qc, N, N_val)
        set_bits(qc, B, 3)

        multiply_mod_fixed(qc, N, X, B, AUX)

        qc.measure(B, range(n))
        counts = self.get_counts_intense(qc)
        measured = int(list(counts.keys())[0], 2)
        self.assertEqual(measured, 6)

    # 12. Test multiply_mod_fixed_power_2_k
    def test_multiply_mod_fixed_power_2_k(self):
        """Test multiplication by fixed X^(2^k)"""
        n = 4
        N_val = 7
        X = 3
        k = 1

        # Inject N_val
        qa_module.N_val = N_val

        # Calculate X^(2^k) mod N classically: 3^(2^1) = 3^2 = 9 = 2 mod 7
        # Operation: B <- B * 2 mod 7

        aux_mul_len = 2 * n * n + 8 * n + 4
        aux_len = 3 * n + aux_mul_len
        total_qubits = 2 * n + aux_len
        qc = QuantumCircuit(total_qubits, n)

        N = list(range(0, n))
        B = list(range(n, 2 * n))
        AUX = list(range(2 * n, 2 * n + aux_len))

        set_bits(qc, N, N_val)
        set_bits(qc, B, 3)  # B = 3

        # Result: 3 * 2 = 6 mod 7
        multiply_mod_fixed_power_2_k(qc, N, X, B, AUX, k)

        qc.measure(B, range(n))
        counts = self.get_counts_intense(qc)
        measured = int(list(counts.keys())[0], 2)
        self.assertEqual(measured, 6)

    # 13. Test multiply_mod_fixed_power_Y
    def test_multiply_mod_fixed_power_Y(self):
        """Test multiplication by fixed X^Y where Y is classical integer"""
        n = 4
        N_val = 7
        X = 3
        Y = 2  # 010 (binary)

        # Inject N_val
        qa_module.N_val = N_val

        # Logic: B <- B * (X^Y) mod N
        # X^Y = 3^2 = 9 = 2 mod 7
        # If B=3, Result = 3*2 = 6

        aux_mul_len = 2 * n * n + 8 * n + 4
        aux_fixed_len = 3 * n + aux_mul_len
        aux_len = n + aux_fixed_len
        total_qubits = 3 * n + aux_len  # N, B, Y(unused as reg, but passed logic), AUX

        # Note: The proj2 function signature takes 'Y_val' as classical int,
        # so we don't need a register for Y, but we need registers for N, B, AUX.
        qc = QuantumCircuit(2 * n + aux_len, n)

        N = list(range(0, n))
        B = list(range(n, 2 * n))
        AUX = list(range(2 * n, 2 * n + aux_len))

        set_bits(qc, N, N_val)
        set_bits(qc, B, 3)

        multiply_mod_fixed_power_Y(qc, N, X, B, AUX, Y)

        qc.measure(B, range(n))
        counts = self.get_counts_intense(qc)
        measured = int(list(counts.keys())[0], 2)
        self.assertEqual(measured, 6)


if __name__ == '__main__':
    unittest.main()