import numpy as np
import scipy as sp

class Matrix:
    def __init__(self, row: int, col: int):
        self.row = row
        self.col = col
        self.matrix = np.zeros((row, col))
        self._xdata = np.array([])
        self._ydata = np.array([])

        self.result = np.matrix([])
        self.result = np.zeros((row, col))

    def showMatrix(self):
        print(self.matrix)

    def isSquare(self) -> bool: # Cek matriks simetris
        return self.row == self.col
    
    def isSingular(self) -> bool: # Cek matriks singular
        return self.isSquare() and np.linalg.det(self.matrix) == 0
    
    def isIdentity(self) -> bool: # Cek matriks identitas
        return self.isSquare() and np.array_equal(self.matrix, np.identity(self.row))

    def isInvertable(self) -> bool: # Cek matriks dapat diinverse atau ngga
        return self.isSquare() and self.isSingular()
    
    # Mengubah matriks menjadi data x dan y
    def _convertToXYData(self):

        self._xdata = []
        self._ydata = []

        for i in range(self.row):
            self._xdata.append(self.matrix[i, 0])
            self._ydata.append(self.matrix[i, 1])
        
    def _getTuple(self, op):
        if op == 'row':
            _ = []
            for i in range(self.row):
                _.append(i) 
            tup = tuple(_)
            return tup

        if op == 'col':
            _ = []
            for i in range(self.col):
                _.append(i) 
            tup = tuple(_)
            return tup

class Calculator(Matrix):
    def __init__(self, row, col):
        super().__init__(row, col)
    
    # OPERASI SEDERHANA
    def operasiTambah(self, m2) -> np.matrix:
        result = np.zeros_like(self.matrix, dtype=float)
        for i in range(self.matrix.shape[0]):
            for j in range(self.matrix.shape[1]):
                result[i, j] = self.matrix[i, j] + m2.matrix[i, j]
        return np.matrix(result)

    def operasiKurang(self, m2) -> np.matrix:
        result = np.zeros_like(self.matrix, dtype=float)
        for i in range(self.matrix.shape[0]):
            for j in range(self.matrix.shape[1]):
                result[i, j] = self.matrix[i, j] - m2.matrix[i, j]
        return np.matrix(result)

    def operasiKali(self, m2) -> np.matrix:        
        result = np.zeros((self.matrix.shape[0], m2.matrix.shape[1]), dtype=float)

        for i in range(self.matrix.shape[0]):
            for j in range(m2.matrix.shape[1]):
                for k in range(self.matrix.shape[1]):
                    result[i, j] += self.matrix[i, k] * m2.matrix[k, j]

        return np.matrix(result)

    def operasiTranspose(self) -> np.matrix:
        result = np.zeros((self.matrix.shape[1], self.matrix.shape[0]), dtype=float)
        for i in range(self.matrix.shape[0]):
            for j in range(self.matrix.shape[1]):
                result[j, i] = self.matrix[i, j]
        return np.matrix(result)

    def operasiDeterminan(self) -> np.matrix:
        def determinant_recursive(matrix):
            if matrix.shape[0] == 1:
                return matrix[0, 0]
            if matrix.shape[0] == 2:
                return matrix[0, 0] * matrix[1, 1] - matrix[0, 1] * matrix[1, 0]
            det = 0
            for c in range(matrix.shape[1]):
                sub_matrix = np.delete(np.delete(matrix, 0, axis=0), c, axis=1)
                det += ((-1) ** c) * matrix[0, c] * determinant_recursive(sub_matrix)
            return det
        return determinant_recursive(self.matrix)
    
    def operasiInverse(self) -> np.matrix:
        det = self.operasiDeterminan()
        if det == 0:
            print("Matrix Tidak Dapat DiInverse! Determinan = 0 (Singular Matrix)")
            return None
        else:
            n = self.matrix.shape[0]
            identity = np.eye(n)
            result = np.zeros_like(self.matrix, dtype=float)
            for i in range(n):
                result[:, i] = np.linalg.solve(self.matrix, identity[:, i])
            return np.matrix(result)

    # OPERASI LINEAR
    def operasiGaussJordan(self, b: np.matrix) -> np.matrix:
        # Inisialisasi augmented matrix [A | b]
        n = self.row
        ab = np.hstack((self.matrix.astype(float), b.astype(float)))
        for k in range(n):
            # Tukar baris untuk memastikan pivot maksimum
            max_row = np.argmax(np.abs(ab[k:, k])) + k
            if ab[max_row, k] == 0:
                raise ValueError("Matriks singular, eliminasi tidak dapat dilanjutkan.")
            if max_row != k:
                ab[[k, max_row]] = ab[[max_row, k]]

            # merubah baris supaya elemen diagonal nya jadi 1
            ab[k] /= ab[k, k]

            # Eliminasi elemen di atas dan di bawah elemen diagonal
            for i in range(n):
                if i != k:
                    ab[i] -= ab[i, k] * ab[k]

        # Mengambil solusi x dan matriks a
        x = ab[:, -1]
        a = ab[:, :-1]
        return x, a

    def operasiCramer(self, matrix) -> np.matrix:
        det_A = self.operasiDeterminan()

        n = self.matrix.shape[0]
        x = np.zeros((n, 1))

        for i in range(n):
            A_copy = self.matrix.copy()
            A_copy[:, i] = matrix[
                :, 0
            ]  ## Ganti kolom ke-i di A_copy dengan matriks hasil (matrix)
            det_Ai = np.linalg.det(A_copy)
            x[i, 0] = det_Ai / det_A
        return np.matrix(x)

    def operasiLuDecomposition(self) -> np.matrix:
        n = self.row

        lower = np.zeros((n, n))  # Inisialisasi matriks lower dengan nol
        upper = np.zeros((n, n))  # Inisialisasi matriks upper dengan nol

        for i in range(n):
            # Matriks Upper
            for T in range(i, n):
                sum_upper = 0
                for j in range(i):
                    sum_upper += lower[i][j] * upper[j][T]
                upper[i][T] = self.matrix[i, T] - sum_upper

            # Matriks Lower
            for T in range(i, n):
                if i == T:
                    lower[i][i] = 1  # Diagonal elemen lower selalu 1
                else:
                    sum_lower = 0
                    for j in range(i):
                        sum_lower += lower[T][j] * upper[j][i]
                    lower[T][i] = (self.matrix[T, i] - sum_lower) / upper[i][i]

        lower = np.matrix(lower)
        upper = np.matrix(upper)

        return lower, upper  # Mengembalikan matriks lower dan upper

    def operasiIterasiJacobi(self, b: np.matrix, x0 = None, tol= 1e-5, max_iter = 1000) -> np.matrix:
        # x0=none Jika tidak diberikan (None), maka secara default akan diinisialisasi sebagai vektor nol sejumlah matriks A[0,0,0]
        if not self.isSquare():
            raise ValueError("Matriks harus persegi untuk metode Jacobi.")
        
        n = self.row

        # insiialsiasi x0
        if x0 is None:
            x0 = np.zeros(n)

        x = x0.copy()
        x_new = np.zeros(n)

        # iterasi jacobi
        for _ in range(max_iter):
            for i in range(n):
                sum_j = sum(self.matrix[i, j] * x[j] for j in range(n) if j != i)
                x_new[i] = (b[i] - sum_j) / self.matrix[i, i]

            # presentase_x = 

            # cek konvergensi
            if np.linalg.norm(x_new - x, ord = np.inf) < tol:
                return np.matrix(x_new)

            # nilai x baru
            x = x_new.copy()
            print(f"{x}, {_}")

        print("Metode Jacobi tidak konvergen setelah mencapai iterasi maksimum.")
        return float(x_new)

    
    def _poly_newton_coefficient(self):
        m = self.row
        x = np.copy(self._xdata)
        a = np.copy(self._ydata)

        for k in range(1, m):
            a[k:m] = (a[k:m] - a[k - 1])/(x[k:m] - x[k - 1])

        return a

    def operasiNewtonPolynomial(self, x) -> np.matrix:
        if self.row < 2:
            return None
        
        a = self._poly_newton_coefficient()
        n = self.row - 1
        p = a[n]

        # Rumus Polinomial Newton
        # p = a[n] + (x - xdata[n]) * p

        for k in range(1, n + 1):
            p = a[n - k] + (x - self._xdata[n - k])*p

        # tambah element di matrix
        if self.col > 1:
            new_row = np.zeros(self.col)
            new_row[0] = x  
            new_row[1] = p  
            
            # Resize 
            self.matrix = np.vstack((self.matrix, new_row))
            self.row += 1  
            self.result = self.matrix  
        

        return p


# matrixA = Calculator(3, 3)
# matrixY = Calculator(3, 1)

# matrixA.matrix = np.matrix([[9, 2, 3], 
#                            [1, 12, 9], 
#                            [4, 5, 14]])

# matrixY.matrix = np.matrix([[7], 
#                            [2], 
#                            [1]])

# matrixA.showMatrix()
# matrixY.showMatrix()

# _ = matrixA.operasiIterasiJacobi(matrixY.matrix)
# print(_)

# hasil = np.zeros((3, 1))

# for i in range(3):
#     hasil[i, 0] = _[i]

# print(hasil)

# matrixA.matrix.dot(hasil)