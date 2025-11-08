import sys

EPS = 1e-9

# --------------------------- 입력 ---------------------------
def read_matrix():
    mat = []
    for line in sys.stdin:
        line = line.strip()
        if not line:
            break
        row = list(map(float, line.split()))
        mat.append(row)
    return mat

# --------------------------- 가우스 소거 ---------------------------
def round_small_values(A):
    for i in range(len(A)):
        for j in range(len(A[0])):
            if abs(A[i][j]) < EPS:
                A[i][j] = 0.0

def gaussian_elimination(A):
    n_rows, n_cols = len(A), len(A[0])
    pivot_cols = []
    row = 0

    for col in range(n_cols - 1):
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

    round_small_values(A)
    return A, pivot_cols

def check_inconsistency(A):
    for row in A:
        if all(abs(x) < EPS for x in row[:-1]) and abs(row[-1]) > EPS:
            print("❌ 모순식 발견 → 해 없음")
            return True
    return False

def find_free_variables(A, pivot_cols):
    n_cols = len(A[0]) - 1
    all_vars = set(range(n_cols))
    free_vars = sorted(all_vars - set(pivot_cols))
    if free_vars:
        print(f"자유변수 존재: {', '.join(f'x{v+1}' for v in free_vars)}")

# --------------------------- LU 분해 ---------------------------
def lu_decomposition(A):

    #해당 부분 추가 / 정방행렬 체크
    
    n_rows = len(A)
    n_vars = len(A[0]) - 1
    if n_rows != n_vars:
        return None, None, None, "singular"
    
    #여기까지 추가
    
    n = len(A)
    coeff = [row[:-1] for row in A]
    B = [row[-1] for row in A]

    L = [[0.0]*n for _ in range(n)]
    U = [[0.0]*n for _ in range(n)]

    for i in range(n):
        for k in range(i, n):
            U[i][k] = coeff[i][k] - sum(L[i][j]*U[j][k] for j in range(i))
        for k in range(i, n):
            if abs(U[i][i]) < EPS:
                return None, None, None, "singular"
            if i == k:
                L[i][i] = 1.0
            else:
                L[k][i] = (coeff[k][i] - sum(L[k][j]*U[j][i] for j in range(i))) / U[i][i]
    return L, U, B, None

# --------------------------- 전진·후진 대입 ---------------------------
def forward_substitution(L, B):
    n = len(L)
    y = [0.0]*n
    for i in range(n):
        if abs(L[i][i]) < EPS:
            raise ValueError("전진대입 불가 — L 행렬의 대각이 0입니다.")
        s = sum(L[i][j]*y[j] for j in range(i))
        y[i] = (B[i]-s)/L[i][i]
    return y

def backward_substitution(U, y):
    n = len(U)
    x = [0.0]*n
    free_vars = []
    for i in range(n-1, -1, -1):
        if abs(U[i][i]) < EPS:
            free_vars.append(i)
            continue
        s = sum(U[i][j]*x[j] for j in range(i+1, n))
        x[i] = (y[i]-s)/U[i][i]
    if free_vars:
        return x, "infinite", free_vars
    return x, "unique", []

# --------------------------- 일반해 출력 ---------------------------
def print_general_solution(A, pivot_cols):
    n_rows, n_cols = len(A), len(A[0]) - 1
    all_vars = set(range(n_cols))
    free_vars = sorted(all_vars - set(pivot_cols))
    param_names = [f"t{i+1}" for i in range(len(free_vars))]

    print("\n♾️  무수히 많은 해 (일반해):")
    print("자유변수 → " + ", ".join(f"x{fv+1} = {p}" for fv, p in zip(free_vars, param_names)))

    exprs = {fv: param_names[i] for i, fv in enumerate(free_vars)}

    for i in reversed(range(n_rows)):
        pivot_col = None
        for j in range(n_cols):
            if abs(A[i][j]) > EPS:
                pivot_col = j
                break
        if pivot_col is None:
            continue

        rhs = A[i][-1]
        valid_terms = []

        # 오른쪽 항들 중 0 아닌 항만 포함
        for j in range(pivot_col + 1, n_cols):
            coeff = -A[i][j]
            if abs(coeff) < EPS:
                continue
            var_expr = exprs.get(j, f"x{j+1}")
            sign = "+" if coeff > 0 else "-"
            coeff_str = "" if abs(abs(coeff) - 1) < EPS else f"{abs(coeff):.3f}*"
            valid_terms.append(f"{sign} {coeff_str}({var_expr})")

        # ✅ 상수항이 0이면 표시 생략
        expr_parts = []
        if abs(rhs) >= EPS:
            expr_parts.append(f"{rhs:.3f}")
        expr_parts.extend(valid_terms)

        expr = " ".join(expr_parts).strip()

        # ✅ 불필요한 "+ -" 정리
        expr = expr.replace("+ -", "- ")
        expr = expr.replace("- -", "+ ")

        # ✅ 맨 앞에 "+" 남는 경우 제거
        if expr.startswith("+"):
            expr = expr[1:].strip()

        exprs[pivot_col] = expr

    # 출력 정리
    print("\n[일반해 형태]")
    for i in range(n_cols):
        expr = exprs.get(i, f"x{i+1}")
        print(f"x{i+1} = {expr}")

# --------------------------- 실행부 ---------------------------
A = read_matrix()
if not A:
    print("입력된 행렬이 없습니다.")
    sys.exit()

L, U, B, flag = lu_decomposition(A)

# LU 실패 시 → 가우스 소거로 판별
if flag == "singular":
    print("\n⚠️ 0 피벗/비정방 → 가우스 소거로 판별")
    A, pivots = gaussian_elimination(A)
    print("\n[가우스 소거 결과]")
    for r in A:
        print(" ".join(f"{x:8.3f}" if abs(x) >= EPS else "      " for x in r))
    if not check_inconsistency(A):
        print_general_solution(A, pivots)
    sys.exit()

# LU 출력
def print_matrix(M, name):
    print(f"\n[{name}]")
    for r in M:
        print(" ".join(f"{x:8.3f}" for x in r))

print_matrix(L, "L 행렬")
print_matrix(U, "U 행렬")

y = forward_substitution(L, B)
x, status, free_vars = backward_substitution(U, y)

if status == "unique":
    print("\n✅ [유일해]")
    for i, v in enumerate(x, start=1):
        print(f"x{i} = {v:.6f}")
