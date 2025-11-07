import sys

EPS = 1e-9  # 오차 허용 기준

def read_matrix():
    mat = []
    for line in sys.stdin:
        line = line.strip()
        if not line:
            break
        row = list(map(float, line.split()))
        mat.append(row)
    return mat

def round_small_values(A):
    """부동소수 오차로 인한 -0.000 등 보정"""
    for i in range(len(A)):
        for j in range(len(A[0])):
            if abs(A[i][j]) < EPS:
                A[i][j] = 0.0

def gaussian_elimination(A):
    """가우스 소거 수행 + 피벗 위치 반환"""
    n_rows, n_cols = len(A), len(A[0])
    pivot_cols = []
    row = 0

    for col in range(n_cols - 1):
        # 피벗 찾기
        pivot = None
        for r in range(row, n_rows):
            if abs(A[r][col]) > EPS:
                pivot = r
                break
        if pivot is None:
            continue

        A[row], A[pivot] = A[pivot], A[row]
        pivot_cols.append(col)

        pivot_val = A[row][col]
        A[row] = [x / pivot_val for x in A[row]]

        for r in range(n_rows):
            if r != row:
                factor = A[r][col]
                A[r] = [a - factor * b for a, b in zip(A[r], A[row])]

        row += 1
        if row >= n_rows:
            break

    round_small_values(A)  # 소거 후 오차 보정
    return A, pivot_cols

def check_inconsistency(A):
    for row in A:
        if all(abs(x) < EPS for x in row[:-1]) and abs(row[-1]) > EPS:
            print("모순식 발견 → 해 없음")
            return True
    return False

def find_free_variables(A, pivot_cols):
    n_cols = len(A[0]) - 1  # 마지막 열은 상수항
    all_vars = set(range(n_cols))
    free_vars = sorted(all_vars - set(pivot_cols))

    if free_vars:
        print(f"자유변수 존재: {', '.join(f'x{v+1}' for v in free_vars)}")
    else:
        find_static_variables(A, pivot_cols)

def find_static_variables(A, pivot_cols):
    """유일해인 경우 각 변수의 값을 계산해 출력"""
    n_cols = len(A[0]) - 1
    variable = [0.0] * n_cols

    for r in range(len(A)):
        row = A[r]
        rhs = row[-1]  # 마지막 값은 상수항
        for c in range(n_cols):
            if abs(row[c]) > 1e-9:  # 피벗 위치
                variable[c] = rhs
                break

    print("[유일해] 변수 값:")
    for i, v in enumerate(variable):
        print(f"x{i+1} = {v:.6f}")



# ------------------ 실행 ------------------
A = read_matrix()
if not A:
    print("입력된 행렬이 없습니다.")
    sys.exit()

A, pivots = gaussian_elimination(A)
print("\n[가우스 소거 결과]")
for r in A:
    print(" ".join(f"{x:8.3f}" for x in r))

if not check_inconsistency(A):
    find_free_variables(A, pivots)
