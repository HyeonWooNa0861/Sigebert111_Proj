import sys
from tkinter import *
from tkinter import ttk

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
            #print("❌ 모순식 발견 → 해 없음")
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

    n_rows = len(A)
    n_vars = len(A[0]) - 1
    if n_rows != n_vars:
        return None, None, None, "singular"
    
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

def return_general_solution(A, pivot_cols):
    return_string = "[해가 무수히 많음 (일반해)]\n\n"

    n_rows, n_cols = len(A), len(A[0]) - 1
    all_vars = set(range(n_cols))
    free_vars = sorted(all_vars - set(pivot_cols))
    param_names = [f"t{i+1}" for i in range(len(free_vars))]

    return_string += "자유변수 → " + ", ".join(f"x{fv+1} = {p}" for fv, p in zip(free_vars, param_names))

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
    for i in range(n_cols):
        expr = exprs.get(i, f"x{i+1}")
        return_string += f"\nx{i+1} = {expr}"

    return return_string

# ————————————— GUI —————————————
class GUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Matrix Calculator")

        # 중앙 배치용 Main Frame
        self.main_frame = Frame(root)
        self.main_frame.place(relx=0.5, rely=0.5, anchor='center')

        # 행렬 크기 조절
        Label(self.main_frame, text="미지수 개수").grid(row=1, column=0, columnspan=2, pady = 5)
        self.mat_size = IntVar(value=2)
        Spinbox(self.main_frame, from_=1, to=5, textvariable=self.mat_size).grid(row=2, column=0, columnspan=2, padx=5, pady=5)
        Button(self.main_frame, text="크기 적용", command=self.build_matrix_inputs).grid(row=4, column=0, pady=5, columnspan=2)

        # 입력 프레임 생성
        self.frame_coeff = Frame(self.main_frame, padx=10, pady=10, bd=1, relief="solid")
        self.frame_coeff.grid(row=5, column=0, padx=10, pady=5)
        self.frame_const = Frame(self.main_frame, padx=10, pady=10, bd=1, relief="solid")
        self.frame_const.grid(row=5, column=1, padx=10, pady=5)

        # 입력창 저장 리스트
        self.matrix_coeff_entry = []
        self.matrix_const_entry = []

        # 연립방정식의 해 및 L, U 행렬 출력용 프레임
        self.result_frame = Frame(self.main_frame, padx=10, pady=10, bd=1, relief="solid")
        self.result_frame_L = Frame(self.result_frame, padx=10, pady=10, bd=1, relief="solid")
        self.result_frame_U = Frame(self.result_frame, padx=10, pady=10, bd=1, relief="solid")

    def build_matrix_inputs(self):
        # 입력 프레임 초기화
        for widget in self.frame_coeff.winfo_children():
            widget.destroy()
        for widget in self.frame_const.winfo_children():
            widget.destroy()

        self.matrix_coeff_entry.clear()
        self.matrix_const_entry.clear()

        # 계수행렬 입력창 (A)
        Label(self.frame_coeff, text="계수행렬").grid(row=0, column=0, columnspan=self.mat_size.get(), pady=5)
        for i in range(self.mat_size.get()):
            row = []
            for j in range(self.mat_size.get()):
                e = Entry(self.frame_coeff, width=5)
                e.grid(row=i+1, column=j, padx=2, pady=2)
                row.append(e)
            self.matrix_coeff_entry.append(row)

        # 상수벡터 입력창 (B)
        Label(self.frame_const, text="상수벡터").grid(row=0, column=0, columnspan=self.mat_size.get(), pady=5)
        for i in range(self.mat_size.get()):
            row = []
            e = Entry(self.frame_const, width=5)
            e.grid(row=i+1, column=0, padx=2, pady=2)
            row.append(e)
            self.matrix_const_entry.append(row)

        # 결과 출력 버튼
        self.result_button = Button(self.main_frame, text="결과 출력", command=self.print_result)
        self.result_button.grid(row=6, column=0, pady=10, columnspan=2)

    def get_matrix_from_entries(self, entries):
        matrix = []
        for row in entries:
            current_row = []
            for entry in row:
                try:
                    val = float(entry.get())
                    current_row.append(val)
                except ValueError:
                    current_row.append(0) # 입력이 비어있거나 숫자가 아닐 때 0 대입
            matrix.append(current_row)
        return matrix

    def print_result(self):
        # 결과 출력 프레임 초기화
        for widget in self.result_frame.winfo_children():
            widget.destroy()

        self.result_frame.grid(row=7, column=0, columnspan=2, pady=5)

        # 입력 행렬 가져오기
        matrix_coeff = self.get_matrix_from_entries(self.matrix_coeff_entry) # 계수 행렬
        matrix_const = self.get_matrix_from_entries(self.matrix_const_entry) # 상수 벡터

        input_matrix = []
        for i in range(self.mat_size.get()):
            input_matrix.append(matrix_coeff[i] + [matrix_const[i][0]])

        # LU 분해 시도
        L, U, B, flag = lu_decomposition(input_matrix)

        if flag == "singular": # 분해 실패
            output_matrix, pivot = gaussian_elimination(input_matrix)

            # 가우스 소거 결과 출력용 프레임
            self.result_label = Label(self.result_frame, text="가우스 소거 결과")
            self.result_gauss_coeff = Frame(self.result_frame, padx=10, pady=10, bd=1, relief="solid")
            self.result_gauss_const = Frame(self.result_frame, padx=10, pady=10, bd=1, relief="solid")

            self.result_label.grid(row=1, column=0, pady=5, columnspan=2)
            self.result_gauss_coeff.grid(row=2, column=0, pady=5)
            self.result_gauss_const.grid(row=2, column=1, pady=5)

            if not check_inconsistency(output_matrix): # 모순식 없음
                return_string = return_general_solution(output_matrix, pivot)
                Label(self.result_frame, text=return_string).grid(row=0, column=0, columnspan=2, pady=5)
            else:
                Label(self.result_frame, text="[해 없음 (모순식 존재)]").grid(row=0, column=0, columnspan=2, pady=5)

            # 가우스 소거 결과 출력
            for i in range(self.mat_size.get()):
                for j in range(self.mat_size.get()):
                    e = Entry(self.result_gauss_coeff, width=8, justify="center")
                    e.grid(row=i+1, column=j, padx=2, pady=2)
                    e.insert(0, f"{output_matrix[i][j]:8.3f}")

            for i in range(self.mat_size.get()):
                e = Entry(self.result_gauss_const, width=8, justify="center")
                e.grid(row=i+1, column=0, padx=2, pady=2)
                e.insert(0, f"{output_matrix[i][-1]:8.3f}")
            
        else:
            # 연립방정식의 해 및 L, U 행렬 출력용 프레임
            self.result_label = Label(self.result_frame, text="[연립방정식의 해]")
            self.result_frame_L = Frame(self.result_frame, padx=10, pady=10, bd=1, relief="solid")
            self.result_frame_U = Frame(self.result_frame, padx=10, pady=10, bd=1, relief="solid")

            # LU 분해 후 해 출력
            self.result_label.grid(row=0, column=0, pady=5, columnspan=2)
            solution = '\n'.join(f"x{i+1} = {val if abs(val)>=EPS else 0:8.3f}  " for i, val in enumerate(backward_substitution(U, forward_substitution(L, B))[0]))
            Label(self.result_frame, text=solution).grid(row=1, column=0, pady=5, columnspan=2)

            Label(self.result_frame, text = "LU 분해 결과").grid(row=2, column=0, columnspan=2, pady=5)
            self.result_frame_L.grid(row=3, column=0, pady=5)
            self.result_frame_U.grid(row=3, column=1, pady=5)

            # L 행렬 출력
            Label(self.result_frame_L, text="L 행렬").grid(row=0, column=0, pady=5, columnspan = self.mat_size.get())
            for i in range(self.mat_size.get()):
                for j in range(self.mat_size.get()):
                    e = Entry(self.result_frame_L, width=8, justify="center")
                    e.grid(row=i+1, column=j, padx=2, pady=2)
                    e.insert(0, f"{L[i][j]:8.3f}" if j <= i else "")

            # U 행렬 출력
            Label(self.result_frame_U, text="U 행렬").grid(row=0, column=0, pady=5, columnspan= self.mat_size.get())
            for i in range(self.mat_size.get()):
                for j in range(self.mat_size.get()):
                    e = Entry(self.result_frame_U, width=8, justify="center")
                    e.grid(row=i+1, column=j, padx=2, pady=2)
                    e.insert(0, f"{U[i][j]:8.3f}" if j >= i else "")
                    
if __name__ == "__main__":
    root = Tk()
    root.title("Matrix Calculator")

    root.geometry("960x1080")
    root.minsize(640, 720)

    gui = GUI(root)
    root.mainloop()
