import numpy as np

# 입력 읽기
n = int(input().strip())
A = np.array([list(map(float, input().split())) for _ in range(n)], dtype=float)
k = int(input().strip())
# 음수 거듭제곱일 경우 대비
k0 = k

# k<0인 경우: A^k = (A^{-1})^{-k} +
if k < 0:
    if np.linalg.det(A) == 0:
        raise ValueError("k<0인데 행렬이 가역이 아니라 A^k를 정의할 수 없습니다.")
    A = np.linalg.inv(A)
    k = -k
# k=0인 경우: 항등행렬
if k == 0:
    # 항등행렬 생성
    result = np.eye(n, dtype=float)
else:
    # 1x1 행렬인 경우 직접 계산
    if n == 1:
        # 단일 원소 A[0,0]의 k제곱
        result = np.array([[A[0,0]**k]])
    else:
        # 행렬의 특성다항식 계수 계산 (고유값 사용)
        eigvals = np.linalg.eigvals(A)
        # poly(eigvals) 결과는 [1, c_{n-1}, ..., c_0]
        char_coefs = np.poly(eigvals).real
        # 계수들을 실수로 변환하고 차수 역순(상수항부터) 리스트로 저장
        char_coefs = np.real_if_close(char_coefs)
        char_coefs_desc = char_coefs.tolist()      # 내림차순 계수
        char_coefs_asc = char_coefs_desc[::-1]     # 오름차순 (상수항부터) 

        # 다항식 곱셈 함수 (다항식 리스트는 오름차순 계수로 표현)
        def poly_mul(p, q):
            res = [0]*(len(p)+len(q)-1)
            for i, pi in enumerate(p):
                for j, qj in enumerate(q):
                    res[i+j] += pi * qj
            return res

        # 다항식 p를 char_coefs_asc로 주어진 특성다항식으로 나눈 나머지 계산
        def poly_mod(poly):
            # char_coefs_asc = [c0, c1, ..., c_{n-1}, 1]
            n_deg = len(char_coefs_asc) - 1  # n
            # 차수 d가 n 이상인 항을 반복적으로 대체
            while len(poly) > n_deg:
                coeff = poly[-1]  # 최고차항 계수
                if coeff != 0:
                    # x^n = -(c_{n-1}x^{n-1}+...+c_0) 이므로
                    for j in range(n_deg):
                        poly[-n_deg-1+j] -= coeff * char_coefs_asc[j]
                poly.pop()  # 최고차항 제거
            return poly

        # x^k를 특성다항식으로 나눈 나머지를 빠른 거듭제곱으로 계산
        def poly_pow(base, exp):
            result_poly = [1]  # 상수항 1 (다항식 1)
            while exp > 0:
                if exp % 2 == 1:
                    result_poly = poly_mod(poly_mul(result_poly, base))
                base = poly_mod(poly_mul(base, base))
                exp //= 2
            return result_poly

        # 기본 다항식 x (오름차순 표현: 0 + 1*x)
        base_poly = [0, 1]
        rem_poly = poly_pow(base_poly, k)  # x^k의 나머지 다항식 계수

        # n-1 차 이하까지의 A의 거듭제곱을 미리 계산
        A_pows = [np.eye(n)]
        for i in range(1, len(rem_poly)):
            A_pows.append(A_pows[-1].dot(A))

        # 남은 다항식 계수(rem_poly)로 행렬 계산: c0*I + c1*A + ... + c_{n-1}*A^{n-1}
        result = np.zeros((n, n))
        for power, coef in enumerate(rem_poly):
            if coef != 0:
                result += coef * A_pows[power]

# 결과 출력 (numpy.array2string 사용, floatmode='fixed'로 소수점 4자리)
result_str = np.array2string(result, precision=4, floatmode='fixed',
                             separator=', ', max_line_width=1000)
# 줄바꿈 제거 (1행으로 출력) 및 '-0.0000'을 '0.0000'으로 치환
result_str = result_str.replace('-0.0000', '0.0000')
print(f"A^{k0} = {result_str}")
