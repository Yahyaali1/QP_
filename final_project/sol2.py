"""
Quantum Arithmetic Library
Extracted from proj2.ipynb
"""

from qiskit import QuantumCircuit
from qiskit.circuit.library import MCXGate

# Global variable to hold N_val for classical pre-computations in fixed multiplication
N_val = None


# --- 1.1 Initialization ---
def set_bits(circuit, A, X):
    """
    Initializes the bits of register A with the binary string or integer X.
    """
    w = len(A)

    if isinstance(X, int):
        if X < 0:
            raise ValueError("X must be a positive integer.")
        X = bin(X)[2:].zfill(w)
    elif isinstance(X, str):
        if X.startswith("0b"):
            X = X[2:].zfill(w)
        else:
            X = X.zfill(w)
    else:
        raise TypeError("X must be an int or str")

    if len(X) > w:
        raise ValueError("Binary value doesn't fit in target register")

    X = X.zfill(w)
    # Reverse string so A[0] is the least significant bit
    X = X[::-1]

    for i in range(w):
        if X[i] == "1":
            circuit.x(A[i])

    circuit.barrier()
    return circuit


# --- 1.2 Copy ---
def copy(circuit, A, B):
    """
    Copies the binary string from register A to register B using CNOT gates.
    Assume B is initialized to |0>.
    """
    for i in range(len(A)):
        circuit.cx(A[i], B[i])
    return circuit


# --- 1.3 Full Adder ---
def full_adder(circuit, a, b, r, c_in, c_out, AUX):
    """
    Implements a full adder.
    """
    # Sum with XOR: r <- r ^ a ^ b ^ c_in
    circuit.cx(a, r)
    circuit.cx(b, r)
    circuit.cx(c_in, r)

    # Carry with CCNOT
    circuit.ccx(a, b, c_out)
    circuit.ccx(a, c_in, c_out)
    circuit.ccx(b, c_in, c_out)

    return circuit


# --- 1.4 Addition ---
def add(circuit, A, B, R, AUX):
    """
    Adds number(A) to number(B) and stores result in R.
    """
    n = len(A)
    if len(B) != n or len(R) != n:
        raise ValueError("len(A), len(B), and len(R) must be the same length")
    if len(AUX) < n + 1:
        raise ValueError("AUX must have at least len(A) + 1 qubits.")

    # Computing sum
    for i in range(len(A)):
        c_in = AUX[i]
        c_out = AUX[i + 1]
        full_adder(circuit, A[i], B[i], R[i], c_in, c_out, AUX)

    # Resetting AUX (uncompute carries)
    for i in range(n - 1, -1, -1):
        c_in = AUX[i]
        c_out = AUX[i + 1]
        circuit.ccx(B[i], c_in, c_out)
        circuit.ccx(A[i], c_in, c_out)
        circuit.ccx(A[i], B[i], c_out)

    return circuit


# --- 1.5 Subtraction ---
def subtract(circuit, A, B, R, AUX):
    """
    Subtracts Number(B) from Number(A) and stores result in R.
    R <- A + !B + 1
    """
    # Negate B
    for i in range(len(B)):
        circuit.x(B[i])

    # Set carry-in to 1
    circuit.x(AUX[0])

    # Add negated B
    add(circuit, A, B, R, AUX)

    # Undo carry-in
    circuit.x(AUX[0])

    # Un-negate B
    for i in range(len(B)):
        circuit.x(B[i])

    return circuit


# --- 1.6 Comparison ---
def greater_or_eq(circuit, A, B, r, AUX):
    """
    Tests if number(A) >= number(B). Result stored in r.
    """
    n = len(A)
    if len(B) != n:
        raise ValueError("len(A) must equal len(B)")
    if len(AUX) < n + 1:
        raise ValueError("AUX register too small.")

    A = A[:n]
    B = B[:n]
    AUX = AUX[:n + 1]

    # Complement B
    for i in range(n):
        circuit.x(B[i])

    # Set initial carry-in to 1
    circuit.x(AUX[0])

    # Compute carry chain
    for i in range(n):
        c_in = AUX[i]
        c_out = AUX[i + 1]
        circuit.ccx(A[i], B[i], c_out)
        circuit.ccx(A[i], c_in, c_out)
        circuit.ccx(B[i], c_in, c_out)

    # Check final carry out
    circuit.cx(AUX[n], r)

    # Uncompute carries
    for i in range(n - 1, -1, -1):
        c_in = AUX[i]
        c_out = AUX[i + 1]
        circuit.ccx(B[i], c_in, c_out)
        circuit.ccx(A[i], c_in, c_out)
        circuit.ccx(A[i], B[i], c_out)

    circuit.x(AUX[0])

    # Reverse complement of B
    for i in range(n):
        circuit.x(B[i])

    return circuit


# --- Helper functions for 1.7 Add Mod ---
def carry_forward(circuit, a, b, aux_carry):
    circuit.cx(aux_carry, b)
    circuit.cx(aux_carry, a)
    circuit.ccx(a, b, aux_carry)


def carry_backward(circuit, a, b, aux_carry):
    circuit.ccx(a, b, aux_carry)
    circuit.cx(aux_carry, a)
    circuit.cx(a, b)


def add_update(circuit, A, B, aux_carry):
    n = len(A)
    for i in range(n):
        carry_forward(circuit, A[i], B[i], aux_carry)
    for i in range(n - 1, -1, -1):
        carry_backward(circuit, A[i], B[i], aux_carry)
    return circuit


# Controlled Helpers
num_controls = 3
mcx_gate = MCXGate(num_controls)


def carry_forward_controlled(circuit, a, b, aux_carry, flag_control):
    circuit.ccx(flag_control, aux_carry, b)
    circuit.ccx(flag_control, aux_carry, a)
    circuit.append(mcx_gate, [flag_control, a, b, aux_carry])


def carry_backward_controlled(circuit, a, b, aux_carry, flag_control):
    circuit.append(mcx_gate, [flag_control, a, b, aux_carry])
    circuit.ccx(flag_control, aux_carry, a)
    circuit.ccx(flag_control, a, b)


def add_update_controlled(circuit, A, B, aux_carry, flag_control):
    n = len(A)
    for i in range(n):
        carry_forward_controlled(circuit, A[i], B[i], aux_carry, flag_control)
    for i in range(n - 1, -1, -1):
        carry_backward_controlled(circuit, A[i], B[i], aux_carry, flag_control)
    return circuit


def subtract_update_controlled(circuit, R, N, flag_compare, aux_carry):
    n = len(R)
    # Temporarily negate N
    for i in range(n):
        circuit.cx(flag_compare, N[i])
    # Controlled +1 on carry
    circuit.cx(flag_compare, aux_carry)
    # R <- R - N (mod 2**n)
    add_update_controlled(circuit, N, R, aux_carry, flag_compare)
    # Undo
    circuit.cx(flag_compare, aux_carry)
    for i in range(n):
        circuit.cx(flag_compare, N[i])
    return circuit


# --- 1.7 Addition Modulo N ---
def add_mod(circuit, N, A, B, R, aux):
    n = len(A)
    if len(B) != n or len(N) != n or len(R) != n:
        raise ValueError("All registers must be the same length")
    if len(aux) < (2 * n + 2):
        raise ValueError("aux must have at least 2n + 2 qubits")

    AUX = aux[: n + 1]
    flag_compare = aux[n + 1]
    aux_carry = AUX[0]
    temp_R = aux[n + 2: n + 2 + n]

    # 1. R <- A + B
    add(circuit, A, B, R, AUX)

    # 2. temp_R <- R
    copy(circuit, R, temp_R)

    # 3. flag <- (temp_R >= N)
    greater_or_eq(circuit, temp_R, N, flag_compare, AUX)

    # 4. If flag: R <- R - N
    subtract_update_controlled(circuit, R, N, flag_compare, aux_carry)

    # 5. Uncompute flag
    greater_or_eq(circuit, temp_R, N, flag_compare, AUX)

    # 6. Clear temp_R
    add(circuit, A, B, temp_R, AUX)

    return circuit


def add_mod_inv(circuit, N, A, B, R, aux):
    """Inverse of add_mod"""
    n = len(A)
    AUX = aux[: n + 1]
    flag_compare = aux[n + 1]
    aux_carry = AUX[0]
    temp_R = aux[n + 2: n + 2 + n]

    # Inverse steps 6..1
    add(circuit, A, B, temp_R, AUX)
    greater_or_eq(circuit, temp_R, N, flag_compare, AUX)
    # Inverse subtraction is addition
    add_update_controlled(circuit, N, R, aux_carry, flag_compare)
    greater_or_eq(circuit, temp_R, N, flag_compare, AUX)
    copy(circuit, R, temp_R)
    add(circuit, A, B, R, AUX)
    return circuit


# --- 1.8 Multiplication by Two Modulo N ---
def times_two_mod(circuit, N, A, R, AUX):
    n = len(A)
    if len(N) != n or len(R) != n:
        raise ValueError("All registers must have same length")

    temp_A = AUX[:n]
    add_aux = AUX[n:]

    # temp_A <- A
    copy(circuit, A, temp_A)

    # R <- (A + temp_A) mod N
    add_mod(circuit, N, A, temp_A, R, add_aux)

    # Uncompute temp_A
    copy(circuit, A, temp_A)

    return circuit


def times_two_mod_inv(circuit, N, A, R, aux):
    """Inverse of times_two_mod"""
    n = len(A)
    temp_A = aux[:n]
    aux_add = aux[n: n + (2 * n + 2)]

    copy(circuit, A, temp_A)
    add_mod_inv(circuit, N, A, temp_A, R, aux_add)
    copy(circuit, A, temp_A)
    return circuit


# --- 1.9 Multiplication by Power of Two Modulo N ---
def times_two_power_mod(circuit, N, A, k, R, AUX):
    if k < 0: raise ValueError("k must be positive")

    n = len(A)
    chain_len = (k + 1) * n
    aux_two_len = 3 * n + 2

    chain_bits = AUX[:chain_len]
    aux_two_mod = AUX[chain_len: chain_len + aux_two_len]

    X_regs = []
    for i in range(k + 1):
        X_regs.append(chain_bits[i * n: (i + 1) * n])

    # X0 <- A
    copy(circuit, A, X_regs[0])

    # Forward
    for i in range(k):
        times_two_mod(circuit, N, X_regs[i], X_regs[i + 1], aux_two_mod)

    # Output
    copy(circuit, X_regs[k], R)

    # Backward
    for i in range(k - 1, -1, -1):
        times_two_mod_inv(circuit, N, X_regs[i], X_regs[i + 1], aux_two_mod)

    # Clear X0
    copy(circuit, A, X_regs[0])

    return circuit


# --- 1.10 Multiplication Modulo N ---
def mask_term(circuit, control_bit, X, Y):
    for i in range(len(X)):
        circuit.ccx(control_bit, X[i], Y[i])
    return circuit


def unmask_term(circuit, control_bit, X, Y):
    return mask_term(circuit, control_bit, X, Y)


def multiply_mod(circuit, N, A, B, R, AUX):
    n = len(A)
    chain_len = (n + 1) * n

    chain_bits = AUX[:chain_len]
    S_regs = [chain_bits[i * n:(i + 1) * n] for i in range(n + 1)]

    temp_aux = AUX[chain_len: chain_len + n]
    temp_mask = AUX[chain_len + n: chain_len + 2 * n]

    k_max = n - 1
    aux_term_len = (k_max + 1) * n + (3 * n + 2)
    AUX_term = AUX[chain_len + 2 * n: chain_len + 2 * n + aux_term_len]

    aux_add_len = 2 * n + 2
    AUX_add = AUX[chain_len + 2 * n + aux_term_len: chain_len + 2 * n + aux_term_len + aux_add_len]

    # Forward pass
    for k in range(n):
        times_two_power_mod(circuit, N, A, k, temp_aux, AUX_term)
        mask_term(circuit, B[k], temp_aux, temp_mask)
        add_mod(circuit, N, S_regs[k], temp_mask, S_regs[k + 1], AUX_add)
        unmask_term(circuit, B[k], temp_aux, temp_mask)
        times_two_power_mod(circuit, N, A, k, temp_aux, AUX_term)
        circuit.barrier()

    copy(circuit, S_regs[n], R)

    # Backward pass
    for k in range(n - 1, -1, -1):
        times_two_power_mod(circuit, N, A, k, temp_aux, AUX_term)
        mask_term(circuit, B[k], temp_aux, temp_mask)
        add_mod_inv(circuit, N, S_regs[k], temp_mask, S_regs[k + 1], AUX_add)
        unmask_term(circuit, B[k], temp_aux, temp_mask)
        times_two_power_mod(circuit, N, A, k, temp_aux, AUX_term)
        circuit.barrier()

    return circuit


# --- 1.11 Multiplication Modulo N (Fixed Factor) ---
def set_const_bits(circuit, reg, value_int):
    n = len(reg)
    bits = bin(value_int)[2:].zfill(n)[::-1]
    for i in range(n):
        if bits[i] == "1":
            circuit.x(reg[i])
    return circuit


def clear_const_bits(circuit, reg, value_int):
    return set_const_bits(circuit, reg, value_int)


def swap_registers(circuit, X, Y):
    for i in range(len(X)):
        circuit.cx(X[i], Y[i])
        circuit.cx(Y[i], X[i])
        circuit.cx(X[i], Y[i])
    return circuit


def multiply_mod_fixed(circuit, N, X, B, AUX):
    n = len(B)
    try:
        X_inv = pow(X, -1, N_val)
    except NameError:
        raise NameError("Global variable N_val must be set before calling multiply_mod_fixed")
    except ValueError:
        raise ValueError("X has no modular inverse mod N_val")

    A_const = AUX[0:n]
    Ainv_const = AUX[n:2 * n]
    R_work = AUX[2 * n:3 * n]
    AUX_mul = AUX[3 * n:]

    set_const_bits(circuit, A_const, X)
    set_const_bits(circuit, Ainv_const, X_inv)

    multiply_mod(circuit, N, A_const, B, R_work, AUX_mul)
    swap_registers(circuit, B, R_work)
    multiply_mod(circuit, N, Ainv_const, B, R_work, AUX_mul)

    clear_const_bits(circuit, Ainv_const, X_inv)
    clear_const_bits(circuit, A_const, X)
    return circuit


# --- 1.12 Multiplication by X^(2^k) mod N ---
def pow2k_mod(W, N_val, k):
    x = W % N_val
    for _ in range(k):
        x = (x * x) % N_val
    return x


def multiply_mod_fixed_power_2_k(circuit, N, X, B, AUX, k):
    try:
        _ = N_val
    except NameError:
        raise NameError("Global variable N_val must be set")

    W = pow2k_mod(X, N_val, k)
    return multiply_mod_fixed(circuit, N, W, B, AUX)


# --- 1.13 Multiplication by X^Y mod N ---
def multiply_mod_fixed_power_Y(circuit, N, X, B, AUX, Y_val):
    n = len(B)
    for k in range(n):
        if (Y_val >> k) & 1:
            multiply_mod_fixed_power_2_k(circuit, N, X, B, AUX, k)
    return circuit