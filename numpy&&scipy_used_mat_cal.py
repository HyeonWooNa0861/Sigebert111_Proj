import numpy as np
from scipy.linalg import fractional_matrix_power

# Read the matrix size
try:
    n = int(input("Enter matrix size n: ").strip())
    if n <= 0:
        print("Error: Matrix size must be a positive integer.")
        exit()
except Exception:
    print("Error: Invalid input for matrix size.")
    exit()

# Read the matrix A
A = []
print(f"Enter the matrix rows (each with {n} numbers):")
for i in range(n):
    row = input().strip().split()
    if len(row) != n:
        print(f"Error: Row {i+1} does not have {n} elements.")
        exit()
    try:
        A.append([float(x) for x in row])
    except ValueError:
        print(f"Error: Non-numeric value in row {i+1}.")
        exit()
A = np.array(A, dtype=float)

# Read the exponent k
try:
    k = float(input("Enter exponent k: ").strip())
except Exception:
    print("Error: Invalid input for exponent.")
    exit()

# If k is negative, check invertibility (A^{-1} must exist)
if k < 0:
    try:
        _ = np.linalg.inv(A)
    except np.linalg.LinAlgError:
        print("Error: Matrix is not invertible; cannot raise to a negative power.")
        exit()

# Compute A^k using SciPy
try:
    result = fractional_matrix_power(A, k)
except ValueError as e:
    print(f"Error: {e}")
    exit()
except Exception as e:
    print(f"Error computing matrix power: {e}")
    exit()

# Function to format each entry to 3 decimals, with a+bi for complex
def format_entry(x):
    real = np.real(x)
    imag = np.imag(x)
    real_r = round(real, 3)
    imag_r = round(imag, 3)
    if abs(imag_r) < 1e-12:
        # Purely real (within rounding)
        return f"{real_r:.3f}"
    # Format complex as a+bi
    sign = '+' if imag_r >= 0 else ''
    return f"{real_r:.3f}{sign}{imag_r:.3f}i"

# Print the resulting matrix
for row in result:
    formatted_row = ' '.join(format_entry(val) for val in row)
    print(formatted_row)
