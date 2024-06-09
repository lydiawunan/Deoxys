import struct

import torch
from joblib import Parallel, delayed
from tqdm import tqdm


def appx_multiplier4x4_AMA5(A,B):
    S = 0
    if (A == 0) or (B == 0):
        S = 0
    elif (A == 1):
        S = B
    elif (A % 2 == 0) & (A>1):
        if (B < 8):
            S = 0
        else:
            S = 32 * (A/2)
    else:
        if (B < 8):
            S = B
        else:
            S = B + 32 * (A-1)/2
    return S


def appx_multiplier8x8(a,b):

    a0b0 = appx_multiplier4x4_AMA5(int(a[4:8],2), int(b[4:8],2))
    a1b0 = appx_multiplier4x4_AMA5(int(a[0:4],2), int(b[4:8],2))
    a0b1 = appx_multiplier4x4_AMA5(int(a[4:8],2), int(b[0:4],2))
    a1b1 = appx_multiplier4x4_AMA5(int(a[0:4],2), int(b[0:4],2))
    S = (a0b0 + (a1b0 + a0b1)*16 + a1b1*256)
    S = format(int(S), '016b')
    return S


def BF_appx_mul(A, B):
    if (abs(A) < 1e-36) or (abs(B) < 1e-36) or (A == 0) or (B == 0):
        s = 0
    else:
        S = ['0', '00000000', '00000000000000000000000']

        a = ''.join(bin(c).replace('0b', '').rjust(8, '0') for c in struct.pack('!f', A))
        b = ''.join(bin(c).replace('0b', '').rjust(8, '0') for c in struct.pack('!f', B))

        sign_ab = int(a[0]) ^ int(b[0])

        exponent_ab = int(a[1:9], 2) + int(b[1:9], 2) - 127
        if exponent_ab < 1:
            s = 0
        else:
            mantissa_ab = appx_multiplier8x8('1' + a[9:16], '1' + b[9:16])

            if mantissa_ab[0] == '1':
                final_mantissa = mantissa_ab[1:8]
                exponent_ab = exponent_ab + 1
            else:
                final_mantissa = mantissa_ab[2:9]
            S = ["{}".format(sign_ab), format(exponent_ab, '08b'), final_mantissa + '0000000000000000']
            S = ''.join(S)
            s = struct.unpack('!f', struct.pack('!I', int(S, 2)))[0]
    return s


def appx_multiplier24x24(a, b):
    a0 = a[16:24]
    a1 = a[8:16]
    a2 = a[0:8]
    b0 = b[16:24]
    b1 = b[8:16]
    b2 = b[0:8]

    a0b0 = int(appx_multiplier8x8(a0, b0), 2)
    a1b0 = int(appx_multiplier8x8(a1, b0), 2) * 256
    a2b0 = int(appx_multiplier8x8(a2, b0), 2) * 256 * 256
    a0b1 = int(appx_multiplier8x8(a0, b1), 2) * 256
    a1b1 = int(appx_multiplier8x8(a1, b1), 2) * 256 * 256
    a2b1 = int(appx_multiplier8x8(a2, b1), 2) * 256 * 256 * 256
    a0b2 = int(appx_multiplier8x8(a0, b2), 2) * 256 * 256
    a1b2 = int(appx_multiplier8x8(a1, b2), 2) * 256 * 256 * 256
    a2b2 = int(appx_multiplier8x8(a2, b2), 2) * 256 * 256 * 256 * 256

    S = a0b0 + a1b0 + a2b0 + a0b1 + a1b1 + a2b1 + a0b2 + a1b2 + a2b2
    S = format(S, '048b')
    # print(S)

    return S

def matrix_appx_multiply_element(row_A, col_B):
    sum = 0
    for i in range(len(row_A)):
        sum += BF_appx_mul(row_A[i],col_B[i])
    return sum

def BF_appx_mul_matrix(A, B,num_jobs=4 ):
    # get the rows and columns of matrices
    rows_A, cols_A = A.shape
    rows_B, cols_B = B.shape

    assert cols_A == rows_B, "check dim match"

    C = torch.zeros((rows_A, cols_B))

    with Parallel(n_jobs = num_jobs) as parallel:
        results = parallel(
            # delayed(matrix_appx_multiply_element)(A[i, :], B[:, j]) for i in tqdm(range(rows_A)) for j in range(cols_B))
            delayed(matrix_appx_multiply_element)(A[i, :], B[:, j]) for i in range(rows_A)for j in range(cols_B))


    for i in range(rows_A):
        for j in range(cols_B):
            C[i, j] = results[i * cols_B + j]

    return C

def BF_appx_mul_Tensor(A, B, C):
    norm = torch.ones(A.shape[0])
    for i in  range(A.shape[0]):
        norm[i] = BF_appx_mul(A[i], B[i])
        norm[i] = BF_appx_mul(norm[i], C[i])
    return norm