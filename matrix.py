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
        result = np.zeros_like(self.matrix, dtype=float)
        for i in range(self.matrix.shape[0]):
            for j in range(self.matrix.shape[1]):
                result[i, j] = self.matrix[i, j] * m2.matrix[i, j]
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

    def operasiCramer(self) -> np.matrix:
        pass

    def operasiLuDecomposition(self) -> np.matrix:
        n = self.row

        lower = np.zeros((n, n))  # Inisialisasi matriks lower dengan nol
        upper = np.zeros((n, n))  # Inisialisasi matriks upper dengan nol

        for i in range(n):
            # Matriks Upper
            for T in range(i, n):
                # Penjumlahan elemen-elemen lower dan upper
                sum = 0
                for j in range(i):
                    sum += lower[i][j] * upper[j][T]
                upper[i][T] = float(self.matrix[i, T].item() - sum)

            # Matriks Lower
            for T in range(i, n):
                if i == T:
                    lower[i][i] = 1
                else:
                    sum = 0
                    for j in range(i):
                        sum += lower[T][j] * upper[j][i]
                lower[T][i] = (float(self.matrix[T, i].item()) - sum) / upper[i][i]

        lower = np.matrix(lower)
        upper = np.matrix(upper)

        return lower, upper  # Mengembalikan matriks lower dan upper

    def operasiIterasiGaussSeidell(self) -> np.matrix:
        pass

    def operasiIterasiJacobi( self, b: np.matrix, x0 = None, tol = 1*10**(-5), max_iter = 100) -> np.matrix:
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

            # cek konvergensi
            if np.linalg.norm(x_new - x, ord = np.inf) < tol:
                return np.matrix(x_new)

            # nilai x baru
            x = x_new.copy()
            print(x," ",  _)

        print("Metode Jacobi tidak konvergen setelah mencapai iterasi maksimum.")
        return x_new

    
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

        return p

    # OPERASI NON-LINEAR
    def operasiNewtonRaphson(self, f, f_aksen, x0: float, tol: float = 1e-7, max_iter: int = 100) -> float:
        """
        Newton-Raphson method for finding roots of a function.

        Parameters:
        - f: Function for which the root is to be found.
        - f_aksen: Derivative of the function f.
        - x0: Initial guess for the root.
        - tol: Tolerance for the solution.
        - max_iter: Maximum number of iterations.

        Returns:
        - The root of the function f.
        """
        x = x0
        for iteration in range(max_iter):
            fx = f(x)
            dfx = f_aksen(x)

            print(f"Iteration {iteration}: x = {x}, f(x) = {fx}, f'(x) = {dfx}")  # Debugging line

            if abs(dfx) < 1e-12:
                raise ValueError("Derivative near zero. Newton-Raphson method fails.")

            # Update step
            x_new = x - fx / dfx

            # Check for convergence
            if abs(x_new - x) < tol:
                print(f"Converged to root: {x_new} after {iteration + 1} iterations")  # Debugging line
                return x_new

            x = x_new

        raise ValueError("Maximum iterations reached. Newton-Raphson method did not converge.")

    def operasiFixedPoint(self) -> np.matrix:
        pass

    def operasiBisection(self) -> np.matrix:
        pass


# UNTUK DEBUGING
# matriks1 = Calculator(3, 2)
# matriks2 = Calculator(2, 2)
matriks3 = Calculator(3, 3)
# matriks4 = Calculator(4, 2)

matriksY = Calculator(3, 1)
# x0_ = Calculator(3, 1)


# matriks1.matrix = np.matrix([[1, 2], [3, 4], [5, 6]])
# matriks2.matrix = np.matrix([[5, 6], [7, 8]])
matriks3.matrix = np.matrix([[9, 3, 3], [1, 12, 9], [4, 6, 14]])
# matriks4.matrix = np.matrix([[1, 5], [2, 4], [3, 3], [4, 2]])

matriksY.matrix = np.matrix([[7], [2], [1]])
# x0 = np.matrix([[0], [0], [0]])

# print('Matriks 1:')
# print(matriks1.matrix)

# print('Matriks 2:')
# print(matriks2.matrix)

# === TEST TAMBAH ===
# matriks1.showMatrix()
# print(matriks1.operasiTambah(matriks2))
# matriks1.showMatrix()

# === TEST KURANG ===
# matriks1.showMatrix()
# print(matriks1.operasiKurang(matriks2))
# matriks1.showMatrix()

# === TEST KALI ===
# matriks1.showMatrix()
# print(matriks1.operasiKali(matriks2))
# matriks1.showMatrix()

# === TEST BAGI ===
# matriks1.showMatrix()
# print(matriks1.operasiBagi(matriks2))
# matriks1.showMatrix()

# === TEST TRANSPOSE ===
# matriks1.showMatrix()
# print(matriks1.operasiTranspose())
# matriks1.showMatrix()

# === TEST INVERSE ===
# matriks1.showMatrix()
# print(matriks1.operasiInverse())
# matriks1.showMatrix()

# === TEST DETERMINAN ===
# matriks1.showMatrix()
# print(matriks1.operasiDeterminan())
# matriks1.showMatrix()

# === TEST GAUS ===
# print("Matriks Awal:")
# solution, tranformed_a = matriks3.operasiGaussJordan(matriksY.matrix)
# print("Solusi x:\n", solution)
# print("Matriks 3 setelah eliminasi Gauss-Jordan:\n", tranformed_a)

# === TEST LU DECOMPOSITION ===
# print("Matriks Awal:")
# matriks3.showMatrix()
# lower, upper = matriks3.operasiLuDecomposition()

# print("Lower Matrix:")
# print(lower)

# print("Upper Matrix:")
# print(upper)

# === TEST INTERPOLASI NEWTON ===
# matriks4._convertToXYData()
# print(matriks4.operasiNewtonPolynomial(8))

# print(matriks4.operasiNewtonPolynomial(2.5))

# === Iterasi Jacobi ===
# print("matriks 3:")
# matriks3.showMatrix()
# print("matriks b:")
# matriksY.showMatrix()
# solusi_jacobi = matriks3.operasiIterasiJacobi(matriksY.matrix)
# print("Solusi dengan metode Jacobi:")
# print(solusi_jacobi)