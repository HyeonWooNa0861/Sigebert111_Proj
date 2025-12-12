import numpy as np

def matrix_power_general(A, k):
    """
    행렬 A와 실수/정수 지수 k에 대해 A^k를 계산.
    정수 k인 경우 np.linalg.matrix_power 사용.
    실수 k인 경우 고유분해를 통해 계산.
    """
    # 정수 지수인지 확인
    if float(k).is_integer():
        k_int = int(k)
        # 양수 정수인 경우
        if k_int >= 0:
            return np.linalg.matrix_power(A, k_int)
        else:
            # 음수 정수인 경우: 가역성 확인
            if np.linalg.det(A) == 0:
                raise np.linalg.LinAlgError("Matrix is not invertible")
            return np.linalg.matrix_power(A, k_int)
    else:
        # 실수 지수 (비정수)인 경우: 고유분해 사용
        eigenvals, V = np.linalg.eig(A)
        # 음수 지수 시 0이 아닌 고유값만 있어야 함
        if k < 0:
            for val in eigenvals:
                if abs(val) < 1e-12:
                    raise np.linalg.LinAlgError("Matrix is not invertible")
        # D^k 계산
        Dk = np.diag(eigenvals ** k)
        # A^k = V * D^k * V^{-1}
        V_inv = np.linalg.inv(V)
        return V.dot(Dk).dot(V_inv)

def format_matrix(M):
    """
    결과 행렬 M을 다음 규칙으로 문자열로 변환:
      - 각 원소를 소수점 3자리로 반올림
      - 음의 영은 0.000으로 표시
      - 복소수는 a+bi 형태로 표시
      - 행렬은 2차원 구조와 줄바꿈을 유지
    """
    rows = []
    for i in range(M.shape[0]):
        row_strs = []
        for j in range(M.shape[1]):
            val = M[i, j]
            # 복소수 여부 확인
            val_c = complex(val)
            a = val_c.real
            b = val_c.imag
            if abs(b) > 1e-12:
                # 실수 및 허수 부분을 각각 3자리로 포맷
                a_str = f"{a:.3f}"
                b_str = f"{abs(b):.3f}"
                if a_str == "-0.000":
                    a_str = "0.000"
                # 부호 처리
                sign = "+" if b >= 0 else "-"
                s = f"{a_str}{sign}{b_str}i"
            else:
                # 순수 실수인 경우
                s = f"{a:.3f}"
                if s == "-0.000":
                    s = "0.000"
            row_strs.append(s)
        rows.append(" ".join(row_strs))
    # 대괄호와 줄바꿈 추가
    if M.shape[0] == 1:
        out = rows[0]
    else:
        out = rows[0] + "\n"
        for row in rows[1:-1]:
            out += row + "\n"
        out += rows[-1]
    return out

def main():
    # 입력 받기
    n = int(input().strip())
    A = []
    for _ in range(n):
        row = list(map(float, input().split()))
        A.append(row)
    A = np.array(A)
    k_line = input().strip()
    # k가 실수인지 정수인지 판별
    if '.' in k_line or 'e' in k_line or 'E' in k_line:
        k = float(k_line)
    else:
        k = int(k_line)
    # 음수 지수인 경우 가역성 확인
    if k < 0 and np.linalg.det(A) == 0:
        print("Error: 행렬이 가역적이지 않습니다.")
        return
    # 계산 및 출력
    Ak = matrix_power_general(A, k)
    result_str = format_matrix(Ak)
    print(result_str)

if __name__ == "__main__":
    main()
