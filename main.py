import matrix
import tkinter as tk
import customtkinter as ctk
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class App:

    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Matrix Calculator")
        self.root.geometry("850x700")
        self.root.resizable(width=False, height=True)
        self.row_countA = 3  
        self.col_countA = 3
        self.row_countB = 3
        self.col_countB = 3

        # inisialisasi matriks
        self.matrixA = matrix.Calculator(self.row_countA, self.col_countA)
        self.matrixB = matrix.Calculator(self.row_countB, self.col_countB)
        self.matrixC = matrix.Calculator(self.row_countA, 1)
        self.matrixD = matrix.Calculator(self.row_countB, 1)
    
        # Header
        header = ctk.CTkLabel(
            master=self.root,
            text="Matrix Calculator",
            font=ctk.CTkFont(family="Arial", size=20, weight="bold"),
            text_color="black"
        )
        header.pack(pady=5, padx=5)
        
        # Main Frame
        self.main_frame = ctk.CTkFrame(master=self.root)
        self.main_frame.pack(padx=5, pady=2, fill="both", expand=True)

        # Result Frame
        self.result_frame = ctk.CTkFrame(master=self.root, width=140, height=350)
        self.result_frame.columnconfigure((0, 1, 2, 3, 4), weight=1, uniform="column")

        # Frame Matrix A
        self.matrix_frame_A = ctk.CTkFrame(master=self.main_frame)
        self.matrix_frame_A.grid(row=0, column=0, columnspan=2, padx=10, pady=10, sticky="nsew")
        self.matrix_frame_A.columnconfigure((0, 1, 2, 3, 4), weight=1, uniform="column")

        self.matrix_btn_frame_A = ctk.CTkFrame(master=self.main_frame, )
        self.matrix_btn_frame_A.grid(row=1, column=0, columnspan=2, padx=10, pady=10, sticky="nsew")


        # Frame Operasi Sederhana
        self.operation_frame = ctk.CTkFrame(master=self.main_frame, width=140, fg_color="transparent")
        self.operation_frame.grid(row=0, column=2, padx=5, pady=5, sticky="nsew")


        # Frame Matrix B
        self.matrix_frame_B = ctk.CTkFrame(master=self.main_frame)
        self.matrix_frame_B.grid(row=0, column=3, columnspan = 2, padx=10, pady=10, sticky="nsew")
        self.matrix_frame_B.columnconfigure((0, 1, 2, 3, 4), weight=1, uniform="column")
        
        self.matrix_btn_frame_B = ctk.CTkFrame(master=self.main_frame)
        self.matrix_btn_frame_B.grid(row=1, column=3, columnspan = 2, padx=10, pady=10, sticky="nsew")
                

        # Header Matrix
        header_matrixA = ctk.CTkLabel(self.matrix_frame_A, text="Matrix A Input", font=("Arial", 14, "bold"),
                                   fg_color="#3B82F6", text_color="white", anchor="center")
        header_matrixA.grid(row=0, column=0, columnspan=6, sticky="new", pady=5)

        header_matrixB = ctk.CTkLabel(self.matrix_frame_B, text="Matrix B Input", font=("Arial", 14, "bold"),
                                   fg_color="#3B82F6", text_color="white", anchor="center")
        header_matrixB.grid(row=0, column=0, columnspan=6, sticky="new", pady=5)

        _containerA = ctk.CTkFrame(self.matrix_frame_A, fg_color = "transparent")
        _containerA.grid(row=2, column=0, columnspan=5, sticky="nsew", pady=5)
        _containerA.columnconfigure((0, 1, 2, 3, 4), weight=1, uniform="column")


        # Input row x col matrix A
        row_label_A = ctk.CTkLabel(_containerA, text="row", font=("Arial", 12))
        row_label_A.grid(row=0, column=1, padx=5, pady=5)

        self.row_spinbox_A = ctk.CTkEntry(_containerA, width=40)
        self.row_spinbox_A.insert(0, "3")
        self.row_spinbox_A.grid(row=1, column=1, padx=5, pady=5)

        multiply_label_A = ctk.CTkLabel(_containerA, text="x", font=("Arial", 12))
        multiply_label_A.grid(row=1, column=2, padx=5)

        col_label_A = ctk.CTkLabel(_containerA, text="column", font=("Arial", 12))
        col_label_A.grid(row=0, column=3, padx=5, pady=5)

        self.col_spinbox_A = ctk.CTkEntry(_containerA, width=40)
        self.col_spinbox_A.insert(0, "3")
        self.col_spinbox_A.grid(row=1, column=3, padx=5, pady=5)


        # Input row x col matrix B
        _containerB = ctk.CTkFrame(self.matrix_frame_B, fg_color="transparent")
        _containerB.grid(row=2, column=0, columnspan=5, sticky="nsew", pady=5)
        _containerB.columnconfigure((0, 1, 2, 3, 4), weight=1, uniform="column")

        row_label_B = ctk.CTkLabel(_containerB, text="row", font=("Arial", 12))
        row_label_B.grid(row=0, column=1, padx=5, pady=5)

        self.row_spinbox_B = ctk.CTkEntry(_containerB, width=40)
        self.row_spinbox_B.insert(0, "3")
        self.row_spinbox_B.grid(row=1, column=1, padx=5, pady=5)

        multiply_label_B = ctk.CTkLabel(_containerB, text="x", font=("Arial", 12))
        multiply_label_B.grid(row=1, column=2, padx=5)

        col_label_B = ctk.CTkLabel(_containerB, text="column", font=("Arial", 12))
        col_label_B.grid(row=0, column=3, padx=5, pady=5)

        self.col_spinbox_B = ctk.CTkEntry(_containerB, width=40)
        self.col_spinbox_B.insert(0, "3")
        self.col_spinbox_B.grid(row=1, column=3, padx=5, pady=5)


        # Update Button Matrix A
        update_button_A = ctk.CTkButton(self.matrix_frame_A, text="Update", command=self.update_matrix)
        update_button_A.grid(row=3, column=0, columnspan=3, pady=5)

        # Toggle Switch Matrix A
        self.switch_A = ctk.CTkSwitch(self.matrix_frame_A, text="matrix Y", command = lambda: self.switch_event("A"), onvalue=True, offvalue=False)
        self.switch_A.grid(row=3, column=3, columnspan = 2, pady=5)

        self.input_frame_matrixA = ctk.CTkFrame(self.matrix_frame_A, fg_color="transparent")
        self.input_frame_matrixA.grid(row=4, column=0, columnspan=3, pady=5)
        
        
        # Update Button Matrix B
        update_button_B = ctk.CTkButton(self.matrix_frame_B, text="Update", command=self.update_matrix)
        update_button_B.grid(row=3, column=0, columnspan=3, pady=5)

        # Toggle Switch Matrix B
        self.switch_B = ctk.CTkSwitch(self.matrix_frame_B, text="matrix Y", command = lambda: self.switch_event("B"), onvalue=True, offvalue=False)
        self.switch_B.grid(row=3, column=3, columnspan = 2, pady=5)

        self.input_frame_matrixB = ctk.CTkFrame(self.matrix_frame_B, fg_color="transparent")
        self.input_frame_matrixB.grid(row=4, column=0, columnspan=3, pady=5)


        # Frame Matrix Ay
        self.matrix_frame_C = ctk.CTkFrame(master=self.matrix_frame_A, width=100, fg_color="transparent")
        self.matrix_frame_C.grid(row=4, column= self.input_frame_matrixA.grid_info()['column'], columnspan = 5, pady=5, padx=5)

        # Frame Matrix By
        self.matrix_frame_D = ctk.CTkFrame(master=self.matrix_frame_B, width=100, fg_color="transparent")
        self.matrix_frame_D.grid(row=4, column= self.input_frame_matrixB.grid_info()['column'], columnspan = 5, pady=5, padx=5)


        # Input Frame Matrix Ay
        self.input_frame_matrixC = ctk.CTkFrame(self.matrix_frame_C)
        
        # Input Frame Matrix By
        self.input_frame_matrixD = ctk.CTkFrame(self.matrix_frame_D)

        # Buat button diantara frame matrixA dan matrixB
        kali_button = ctk.CTkButton(self.operation_frame, text="A x B", command= lambda: self.verify('x', None), width = 60, height = 25)
        kali_button.pack(side="bottom", padx=5, pady=5)

        kurang_button = ctk.CTkButton(self.operation_frame, text="A - B", command= lambda: self.verify('-', None), width = 60, height = 25)
        kurang_button.pack(side="bottom", padx=5, pady=5)

        tambah_button = ctk.CTkButton(self.operation_frame, text="A + B", command= lambda: self.verify('+', None), width = 60, height = 25)
        tambah_button.pack(side="bottom", padx=5, pady=5)

        # matrix A
        determinan_button = ctk.CTkButton(self.matrix_btn_frame_A, text="Determinan", command= lambda: self.verify('det', 'A'), width = 170, height = 25)
        determinan_button.grid(row=0, column=0, padx=5, pady=5)

        transpose_button = ctk.CTkButton(self.matrix_btn_frame_A, text="Transpose", command= lambda: self.verify('t', 'A'), width = 170, height = 25)
        transpose_button.grid(row=0, column=1, padx=5, pady=5)

        inverse_button = ctk.CTkButton(self.matrix_btn_frame_A, text="Inverse", command= lambda: self.verify('inv', 'A'), width = 170, height = 25)
        inverse_button.grid(row=1, column=0, padx=5, pady=5)
        
        lu_decomposition_button = ctk.CTkButton(self.matrix_btn_frame_A, text="LU Decomposition", command= lambda: self.verify('lu', 'A'), width = 170, height = 25)
        lu_decomposition_button.grid(row=1, column=1, padx=5, pady=5)

        self.gauss_jordanA_button = ctk.CTkButton(self.matrix_btn_frame_A, text="Gauss Jordan", command= lambda: self.verify('gj', 'A'), width = 170, height = 25, state = "disabled")
        self.gauss_jordanA_button.grid(row=2, column=1, padx=5, pady=5)

        self.cramerA_button = ctk.CTkButton(self.matrix_btn_frame_A, text="Cramer", command= lambda: self.verify('cr', 'A'), width = 170, height = 25, state = "disabled")
        self.cramerA_button.grid(row=2, column=0, padx=5, pady=5)

        self.jacobiA_button = ctk.CTkButton(self.matrix_btn_frame_A, text="Jacobi", command= lambda: self.verify('jac', 'A'), width = 170, height = 25, state = "disabled")
        self.jacobiA_button.grid(row=3, column=0, padx=5, pady=5)

        newtonPolinomialA_frame = ctk.CTkFrame(self.matrix_btn_frame_A, width = 170, height = 25, fg_color = "transparent")
        newtonPolinomialA_frame.grid(row=3, column=1, padx=0, pady=0)

        self.newtonPolinomialA_button = ctk.CTkButton(newtonPolinomialA_frame, text="Newton Polynomial", command= lambda: self.verify('np', 'A'), width = 140, height = 25, state= "disabled")
        self.newtonPolinomialA_button.pack(side="left", padx=5, pady=5, fill = "both", expand = True)
        
        self.newtonPolinomialA_entry = ctk.CTkEntry(newtonPolinomialA_frame, placeholder_text="Newton Polynomial", width = 30, height = 25)
        self.newtonPolinomialA_entry.pack(side="right", padx=0, pady=5, fill = "x", expand = True)
        self.newtonPolinomialA_entry.insert(0, 0)

        # matrix B
        determinan_button = ctk.CTkButton(self.matrix_btn_frame_B, text="Determinan", command= lambda: self.verify('det', 'B'), width = 170, height = 25)
        determinan_button.grid(row=0, column=0, padx=5, pady=5)

        transpose_button = ctk.CTkButton(self.matrix_btn_frame_B, text="Transpose", command= lambda: self.verify('t', 'B'), width = 170, height = 25)
        transpose_button.grid(row=0, column=1, padx=5, pady=5)

        inverse_button = ctk.CTkButton(self.matrix_btn_frame_B, text="Inverse", command= lambda: self.verify('inv', 'B'), width = 170, height = 25)
        inverse_button.grid(row=1, column=0, padx=5, pady=5)
        
        lu_decomposition_button = ctk.CTkButton(self.matrix_btn_frame_B, text="LU Decomposition", command= lambda: self.verify('lu', 'B'), width = 170, height = 25)
        lu_decomposition_button.grid(row=1, column=1, padx=5, pady=5)

        self.gauss_jordanB_button = ctk.CTkButton(self.matrix_btn_frame_B, text="Gauss Jordan", command= lambda: self.verify('gj', 'B'), width = 170, height = 25, state = "disabled")
        self.gauss_jordanB_button.grid(row=2, column=1, padx=5, pady=5)

        self.cramerB_button = ctk.CTkButton(self.matrix_btn_frame_B, text="Cramer", command= lambda: self.verify('cr', 'B'), width = 170, height = 25, state = "disabled")
        self.cramerB_button.grid(row=2, column=0, padx=5, pady=5)

        self.jacobiB_button = ctk.CTkButton(self.matrix_btn_frame_B, text="Jacobi", command= lambda: self.verify('jac', 'B'), width = 170, height = 25, state = "disabled")
        self.jacobiB_button.grid(row=3, column=0, padx=5, pady=5)

        newtonPolinomialB_frame = ctk.CTkFrame(self.matrix_btn_frame_B, width = 170, height = 25, fg_color = "transparent")
        newtonPolinomialB_frame.grid(row=3, column=1, padx=0, pady=0)

        self.newtonPolinomialB_button = ctk.CTkButton(newtonPolinomialB_frame, text="Newton Polynomial", command= lambda: self.verify('np', 'B'), width = 140, height = 25, state= "disabled")
        self.newtonPolinomialB_button.pack(side="left", padx=5, pady=5, fill = "both", expand = True)
        
        self.newtonPolinomialB_entry = ctk.CTkEntry(newtonPolinomialB_frame, placeholder_text="Newton Polynomial", width = 30, height = 25)
        self.newtonPolinomialB_entry.pack(side="right", padx=0, pady=5, fill = "x", expand = True)
        self.newtonPolinomialB_entry.insert(0, 0)
        
        # Panggil matrix
        self.create_matrix_entries()

    def switch_event(self, mat):
        if mat == 'A':
            if self.switch_A.get():
                self.matrix_frame_C.grid(row=4, column=self.col_countA, padx = 10, pady = 0)

                # on button (gauss, cramer)
                self.gauss_jordanA_button.configure(state = "normal")
                self.cramerA_button.configure(state = "normal")
                self.jacobiA_button.configure(state = "normal")
            else:
                self.matrix_frame_C.grid_remove() 
                
                # off button (gauss, cramer)
                self.gauss_jordanA_button.configure(state = "disabled")
                self.cramerA_button.configure(state = "disabled")
                self.jacobiA_button.configure(state = "disabled")

        if mat == 'B':
            if self.switch_B.get():
                self.matrix_frame_D.grid(row=4, column=self.col_countA, padx = 10, pady = 0)
                
                # on button (gauss, cramer)
                self.gauss_jordanB_button.configure(state = "normal")
                self.cramerB_button.configure(state = "normal")
                self.jacobiB_button.configure(state = "normal")
            else:
                self.matrix_frame_D.grid_remove()  
                
                # off button (gauss, cramer)
                self.gauss_jordanB_button.configure(state = "disabled")
                self.cramerB_button.configure(state = "disabled")
                self.jacobiB_button.configure(state = "disabled")
                
         
    def create_matrix_entries(self):
        # Hapus entri lama
        for widget in self.input_frame_matrixA.winfo_children():
            widget.destroy()

        for widget in self.input_frame_matrixB.winfo_children():
            widget.destroy()

        for widget in self.matrix_frame_C.winfo_children():
            widget.destroy()

        for widget in self.matrix_frame_D.winfo_children():
            widget.destroy()

        # Buat entri baru sesuai row_count dan col_count matrix A
        self.entriesA = []
        for i in range(self.row_countA):
            row_entries = []
            for j in range(self.col_countA):
                entry = ctk.CTkEntry(self.input_frame_matrixA, width=50)
                entry.grid(row=i, column=j, padx=2, pady=2)
                row_entries.append(entry)
            self.entriesA.append(row_entries)

        # Buat entri baru sesuai row_count dan col_count matrix B
        self.entriesB = []
        for i in range(self.row_countB):
            row_entries = []
            for j in range(self.col_countB):
                entry = ctk.CTkEntry(self.input_frame_matrixB, width=50)
                entry.grid(row=i, column=j, padx=2, pady=2)
                row_entries.append(entry)
            self.entriesB.append(row_entries)

        # Buat entri baru sesuai row_count dan col_count matrix y
        self.entriesC = []
        for i in range(self.row_countA):
            row_entries = []
            for j in range(1):
                entry = ctk.CTkEntry(self.matrix_frame_C, width=50)
                entry.grid(row=i, column= j, padx=2, pady=2)
                row_entries.append(entry)
            self.entriesC.append(row_entries)
        
        # Buat entri baru sesuai row_count dan col_count matrix D
        self.entriesD = []
        for i in range(self.row_countB):
            row_entries = []
            for j in range(1):
                entry = ctk.CTkEntry(self.matrix_frame_D, width=50)
                entry.grid(row=i, column=j, padx=2, pady=2)
                row_entries.append(entry)
            self.entriesD.append(row_entries)

        self.switch_event('A')
        self.switch_event('B')

    def update_matrix(self):
        # Update row_count dan col_count berdasarkan input matrix A
        try:
            self.row_countA = int(self.row_spinbox_A.get())
            self.col_countA = int(self.col_spinbox_A.get())
            self.row_countB = int(self.row_spinbox_B.get())
            self.col_countB = int(self.col_spinbox_B.get())

            self.matrixA.__init__(self.row_countA, self.col_countA)
            self.matrixB.__init__(self.row_countB, self.col_countB)
            self.matrixC.__init__(self.row_countA, 1)
            self.matrixD.__init__(self.row_countB, 1)

            self.create_matrix_entries()

            # on/off button (newton polinomial)
            if self.col_countA == 2:
                self.newtonPolinomialA_button.configure(state = "normal")
            else:
                self.newtonPolinomialA_button.configure(state = "disabled")

            # on/off button (newton polinomial)
            if self.col_countB == 2:
                self.newtonPolinomialB_button.configure(state = "normal")
            else:
                self.newtonPolinomialB_button.configure(state = "disabled")
        except ValueError:
            print("Invalid input! Please enter valid integers for row and column.")

    # Import matrix dari entry ke class
    def verify(self, op:str, mat):
        # Import matrix A
        try: 
            # print('masuk A')
            for i in range(self.row_countA):
                for j in range(self.col_countA):
                    if self.entriesA[i][j].get() == '':
                        self.entriesA[i][j].insert(0, '0')
                    self.matrixA.matrix[i][j] = self.entriesA[i][j].get()
        except:
            Exception('Error')

        # Import matrix B
        try:
            # print('masuk B')
            for i in range(self.row_countB):
                for j in range(self.col_countB):
                    if self.entriesB[i][j].get() == '':
                        self.entriesB[i][j].insert(0, '0')
                    self.matrixB.matrix[i][j] = self.entriesB[i][j].get()
        except:
            Exception('Error')

        # Import matrix y
        try:
            # print('masuk y')
            for i in range(self.row_countA):
                for j in range(1):
                    if self.entriesC[i][j].get() == '':
                        self.entriesC[i][j].insert(0, '0')
                    self.matrixC.matrix[i] = self.entriesC[i][j].get()
        except:
            Exception('Error')
            
        # Import matrix D
        try:
            # print('masuk D')
            for i in range(self.row_countB):
                for j in range(1):
                    if self.entriesD[i][j].get() == '':
                        self.entriesD[i][j].insert(0, '0')
                    self.matrixD.matrix[i][j] = self.entriesD[i][j].get()
        except:
            Exception('Error')

        # print("Hasil:")
        if op == '+':
            self.matrixA.result = self.matrixA.operasiTambah(self.matrixB)
            self.spawn_result("A")
        elif op == '-':
            self.matrixA.result = self.matrixA.operasiKurang(self.matrixB)
            self.spawn_result("A")
        elif op == 'x':
            self.matrixA.result = self.matrixA.operasiKali(self.matrixB)
            self.spawn_result("A")

        if mat == 'A':
            if op == 'det':
                self.matrixA.result = self.matrixA.operasiDeterminan()
                self.spawn_result("A", op = 'det')
                # print(self.matrixA.result)
            elif op == 't':
                self.matrixA.result = self.matrixA.operasiTranspose()
                self.spawn_result("A")
                # print(self.matrixA.result)
            elif op == 'inv':
                self.matrixA.result = self.matrixA.operasiInverse()
                # print(self.matrixA.result)
                self.spawn_result("A")
            elif op == 'gj':
                _output = 3
                var, self.matrixA.result = self.matrixA.operasiGaussJordan(self.matrixC.matrix)
                self.spawn_result("A", _output, variable = var, op = 'gj')
                # print(self.matrixA.matrix)
                # print(f"variabel [x, y, z]:\n{type(var)} \n\nresult: \n{self.matrixA.result}")
            elif op == 'cr':
                _output = 3
                self.matrixA.result = self.matrixA.operasiCramer(self.matrixC.matrix)
                # print(self.matrixA.result)
                self.spawn_result("A", _output, op = 'cr')
            elif op == 'lu':
                _output = 3
                lower , upper = (self.matrixA.operasiLuDecomposition())
                self.spawn_result("A", _output, L=lower, U=upper)
                # print(f"lower:\n{lower} \n\nupper: \n{upper}")
            elif op == 'jac':
                self.matrixA.result = self.matrixA.operasiIterasiJacobi(self.matrixC.matrix)
                print(self.matrixA.result)
                self.spawn_result("A", op = 'jac')
            elif op == 'np':
                self.matrixA._convertToXYData()
                self.matrixA.result = self.matrixA.operasiNewtonPolynomial(float(self.newtonPolinomialA_entry.get()))

                self.matrixA._convertToXYData()
                self.spawn_result("A", op = 'np', xAxes = float(self.newtonPolinomialA_entry.get()))

        elif mat == 'B':
            if op == 'det':
                self.matrixB.result = self.matrixB.operasiDeterminan()
                self.spawn_result("B", op = 'det')
                # print(self.matrixB.result)
            elif op == 't':
                self.matrixB.result = self.matrixB.operasiTranspose()
                self.spawn_result("B")
                # print(self.matrixB.result)
            elif op == 'inv':
                self.matrixB.result = self.matrixB.operasiInverse()
                # print(self.matrixB.result)
                self.spawn_result("B")
            elif op == 'gj':
                _output = 3
                var, self.matrixB.result = self.matrixB.operasiGaussJordan(self.matrixD.matrix)
                self.spawn_result("B", _output, variable = var, op = 'gj')
                # print(self.matrixB.matrix)
                # print(f"variabel [x, y, z]:\n{type(var)} \n\nresult: \n{self.matrixB.result}")
            elif op == 'cr':
                _output = 3
                self.matrixB.result = self.matrixB.operasiCramer(self.matrixD.matrix)
                # print(self.matrixB.result)
                self.spawn_result("B", _output, op = 'cr')
            elif op == 'lu':
                _output = 3
                lower , upper = (self.matrixB.operasiLuDecomposition())
                self.spawn_result("B", _output, L=lower, U=upper)
                # print(f"lower:\n{lower} \n\nupper: \n{upper}")
            elif op == 'jac':
                self.matrixB.result = self.matrixB.operasiIterasiJacobi(self.matrixD.matrix)
                print(self.matrixB.result)
                self.spawn_result("B", op = 'jac')
            elif op == 'np':
                self.matrixB._convertToXYData()
                self.matrixB.result = self.matrixB.operasiNewtonPolynomial(float(self.newtonPolinomialB_entry.get()))

                self.matrixB._convertToXYData()
                self.spawn_result("B", op = 'np', xAxes = float(self.newtonPolinomialB_entry.get()))

    def spawn_result(self, mat, out = 1, L = None, U = None, op = None, variable = None, xAxes = None):

        for widget in self.result_frame.winfo_children():
            widget.destroy()
        
        # header
        result_header = ctk.CTkLabel(self.result_frame,
                                    text = f"Result Matrix {mat}", 
                                    width = 50,
                                    font = ctk.CTkFont(family="Arial", size = 16, weight = "bold"),
                                    text_color = "white",
                                    justify = "left",
                                    anchor = "w"
                                    )
        result_header.pack(side = "top", padx=5, pady=5)

        if mat == 'A' and L is not None and U is not None:
            # result frame
            self.result_frame.pack(padx=5, pady=2, fill="both", expand=True)
            result_matrix_frames = []

            for i in range(out):
                # Result Labels Frame
                resultLabels_frame = ctk.CTkFrame(master=self.result_frame)
                resultLabels_frame.pack(side = "left", padx=5, pady=5, fill="both", expand=True)
                resultLabels_frame.columnconfigure(self.matrixA._getTuple('col'), weight=1, uniform="column")
                resultLabels_frame.rowconfigure(self.matrixA._getTuple('row'), weight=1, uniform="column")
                result_matrix_frames.append(resultLabels_frame)

            resultMatrix_labels = []

            graph_frame = ctk.CTkFrame(master=self.result_frame)
            graph_frame.pack(side = "right", padx=5, pady=5, fill="both", expand=True)
            
            _empty = ctk.CTkLabel(graph_frame, text="No Graph", font = ctk.CTkFont(family="Arial", size = 16, weight = "bold"), text_color = "white")
            _empty.pack(padx = 5, pady = 5, fill = "both", expand = True)

            for i in range(self.row_countA):
                for j in range(self.col_countA):
                    # Result Labels for Matrix L
                    label = ctk.CTkLabel(result_matrix_frames[0],  # Frame for L
                                        text=f"{L[i, j]:.2f}",
                                        font=ctk.CTkFont(family="Arial", size=16, weight="bold"),
                                        text_color="white",
                                        justify="center",
                                        anchor="center")
                    label.grid(row=i, column=j, padx=2, pady=2)

            for i in range(self.row_countA):
                for j in range(self.col_countA):
                    # Result Labels for Matrix U
                    label_U = ctk.CTkLabel(result_matrix_frames[1],  # Frame for U
                                        text=f"{U[i, j]:.2f}",
                                        font=ctk.CTkFont(family="Arial", size=16, weight="bold"),
                                        text_color="white",
                                        justify="center",
                                        anchor="center")
                    label_U.grid(row=i, column=j, padx=2, pady=2)

            for i in range(self.row_countA):
                for j in range(self.col_countA):
                    # Result Labels untuk Matrix A
                    label_A = ctk.CTkLabel(result_matrix_frames[2],  # Frame for A
                                        text=f"{self.matrixA.matrix[i, j]:.2f}",
                                        font=ctk.CTkFont(family="Arial", size=16, weight="bold"),
                                        text_color="white",
                                        justify="center",
                                        anchor="center")
                    label_A.grid(row=i, column=j, padx=2, pady=2)
    
        elif mat == 'A' and op == 'cr':
            # result frame
            self.result_frame.pack(padx=5, pady=2, fill="both", expand=True)
            result_matrix_frames = []

            for i in range(out):
                # Result Labels Frame
                resultLabels_frame = ctk.CTkFrame(master=self.result_frame)
                resultLabels_frame.pack(side = "left", padx=5, pady=5, fill="both", expand=True)
                resultLabels_frame.columnconfigure(self.matrixA._getTuple('col'), weight=1, uniform="column")
                resultLabels_frame.rowconfigure(self.matrixA._getTuple('row'), weight=1, uniform="column")
                result_matrix_frames.append(resultLabels_frame)

            resultMatrix_labels = []

            graph_frame = ctk.CTkFrame(master=self.result_frame)
            graph_frame.pack(side = "right", padx=5, pady=5, fill="both", expand=True)
            
            _empty = ctk.CTkLabel(graph_frame, text="No Graph", font = ctk.CTkFont(family="Arial", size = 16, weight = "bold"), text_color = "white")
            _empty.pack(padx = 5, pady = 5, fill = "both", expand = True)

            for i in range(self.row_countA):
                for j in range(self.col_countA):
                    # Result Labels for Matrix A
                    label = ctk.CTkLabel(result_matrix_frames[0],  # Frame for A
                                        text=f"{self.matrixA.matrix[i, j]:.2f}",
                                        font=ctk.CTkFont(family="Arial", size=16, weight="bold"),
                                        text_color="white",
                                        justify="center",
                                        anchor="center")
                    label.grid(row=i, column=j, padx=2, pady=2)

            for i in range(self.row_countA):
                for j in range(0, 1):
                    # Result Labels for Matrix Variable
                    label_var = ctk.CTkLabel(result_matrix_frames[1],  # Frame for Variable
                                        text=f"{self.matrixA.result[i, j]:.2f}",
                                        font=ctk.CTkFont(family="Arial", size=16, weight="bold"),
                                        text_color="white",
                                        justify="center",
                                        anchor="center")
                    label_var.grid(row=i, column=j, padx=2, pady=2)

            for i in range(self.row_countA):
                for j in range(0, 1):
                    # Result Labels untuk Matrix C
                    label_C = ctk.CTkLabel(result_matrix_frames[2],  # Frame for C
                                        text=f"{self.matrixC.matrix[i, j]:.2f}",
                                        font=ctk.CTkFont(family="Arial", size=16, weight="bold"),
                                        text_color="white",
                                        justify="center",
                                        anchor="center")
                    label_C.grid(row=i, column=j, padx=2, pady=2)

        elif mat == 'A' and op == 'gj':
            # result frame
            self.result_frame.pack(padx=5, pady=2, fill="both", expand=True)
            result_matrix_frames = []

            for i in range(out):
                # Result Labels Frame
                resultLabels_frame = ctk.CTkFrame(master=self.result_frame)
                resultLabels_frame.pack(side = "left", padx=5, pady=5, fill="both", expand=True)
                resultLabels_frame.columnconfigure(self.matrixA._getTuple('col'), weight=1, uniform="column")
                resultLabels_frame.rowconfigure(self.matrixA._getTuple('row'), weight=1, uniform="column")
                result_matrix_frames.append(resultLabels_frame)

            resultMatrix_labels = []

            graph_frame = ctk.CTkFrame(master=self.result_frame)
            graph_frame.pack(side = "right", padx=5, pady=5, fill="both", expand=True)
            
            _empty = ctk.CTkLabel(graph_frame, text="No Graph", font = ctk.CTkFont(family="Arial", size = 16, weight = "bold"), text_color = "white")
            _empty.pack(padx = 5, pady = 5, fill = "both", expand = True)

            for i in range(self.row_countA):
                for j in range(self.col_countA):
                    # Result Labels for Matrix A
                    label = ctk.CTkLabel(result_matrix_frames[0],  # Frame for A
                                        text=f"{self.matrixA.matrix[i, j]:.2f}",
                                        font=ctk.CTkFont(family="Arial", size=16, weight="bold"),
                                        text_color="white",
                                        justify="center",
                                        anchor="center")
                    label.grid(row=i, column=j, padx=2, pady=2)

            for i in range(self.row_countA):
                for j in range(0, 1):
                    # Result Labels untuk Matrix Hasil
                    label_result = ctk.CTkLabel(result_matrix_frames[2],  # Frame for Hasil
                                        text=f"{self.matrixA.result[i, j]:.2f}",
                                        font=ctk.CTkFont(family="Arial", size=16, weight="bold"),
                                        text_color="white",
                                        justify="center",
                                        anchor="center")
                    label_result.grid(row=i, column=j, padx=2, pady=2)

            for i in range(self.row_countA):
                # Result Labels for Matrix Variable
                label_var = ctk.CTkLabel(result_matrix_frames[1],  # Frame for Variable
                                    text=f"{variable[i]:.2f}",
                                    font=ctk.CTkFont(family="Arial", size=16, weight="bold"),
                                    text_color="white",
                                    justify="center",
                                    anchor="center")
                label_var.grid(row=i, column=j, padx=2, pady=2)

        elif mat == 'A' and op == 'np':
            # result frame
            self.result_frame.pack(padx=5, pady=2, fill="both", expand=True)
            result_matrix_frames = []

            for i in range(out):
                # Result Labels Frame
                resultLabels_frame = ctk.CTkFrame(master=self.result_frame)
                resultLabels_frame.pack(side = "left", padx=5, pady=5, fill="both", expand=True)
                resultLabels_frame.columnconfigure(self.matrixA._getTuple('col'), weight=1, uniform="column")
                resultLabels_frame.rowconfigure(self.matrixA._getTuple('row'), weight=1, uniform="column")
                result_matrix_frames.append(resultLabels_frame)

            resultMatrix_labels = []

            graph_frame = ctk.CTkFrame(master=self.result_frame)
            graph_frame.pack(side = "right", padx=5, pady=5, fill="both", expand=True)
           
            # Buat plot grafik
            self.plot_frame = ctk.CTkFrame(graph_frame)
            self.plot_frame.pack(fill="both", expand=True, padx=10, pady=10)

            # Create a plot
            self.create_plot("A")

            for i in range(self.matrixA.matrix.shape[0] + 1):
                for j in range(self.col_countA):
                    # Result Labels for Matrix A
                    label = ctk.CTkLabel(result_matrix_frames[0],  # Frame for A
                                        text=f"{self.matrixA.matrix[i, j]:.2f}",
                                        font=ctk.CTkFont(family="Arial", size=16, weight="bold"),
                                        text_color="white",
                                        justify="center",
                                        anchor="center")
                    label.grid(row=i, column=j, padx=2, pady=2)

        elif mat == 'A' and op == 'jac':
            # result frame
            self.result_frame.pack(padx=5, pady=2, fill="both", expand=True)
            result_matrix_frames = []

            for i in range(out):
                # Result Labels Frame
                resultLabels_frame = ctk.CTkFrame(master=self.result_frame)
                resultLabels_frame.pack(side = "left", padx=5, pady=5, fill="both", expand=True)
                resultLabels_frame.columnconfigure(self.matrixA._getTuple('col'), weight=1, uniform="column")
                resultLabels_frame.rowconfigure(self.matrixA._getTuple('row'), weight=1, uniform="column")
                result_matrix_frames.append(resultLabels_frame)

            resultMatrix_labels = []

            graph_frame = ctk.CTkFrame(master=self.result_frame)
            graph_frame.pack(side = "right", padx=5, pady=5, fill="both", expand=True)
            
            _empty = ctk.CTkLabel(graph_frame, text="No Graph", font = ctk.CTkFont(family="Arial", size = 16, weight = "bold"), text_color = "white")
            _empty.pack(padx = 5, pady = 5, fill = "both", expand = True)

            for i in range(len(self.matrixA.result)):
                label_result = ctk.CTkLabel(result_matrix_frames[0],  # Frame for Hasil
                                        text=f"{self.matrixA.result[i]}",
                                        font=ctk.CTkFont(family="Arial", size=16, weight="bold"),
                                        text_color="white",
                                        justify="center",
                                        anchor="center"
                                        )
                label_result.pack(side = "left", padx = 5, pady = 5, fill = "both", expand = True)
                resultMatrix_labels.append(label_result)

        elif mat == 'A' and op == 'det':
            # result frame
            self.result_frame.pack(padx=5, pady=2, fill="both", expand=True)
            result_matrix_frames = []

            for i in range(out):
                # Result Labels Frame
                resultLabels_frame = ctk.CTkFrame(master=self.result_frame)
                resultLabels_frame.pack(side = "left", padx=5, pady=5, fill="both", expand=True)
                resultLabels_frame.columnconfigure(self.matrixA._getTuple('col'), weight=1, uniform="column")
                resultLabels_frame.rowconfigure(self.matrixA._getTuple('row'), weight=1, uniform="column")
                result_matrix_frames.append(resultLabels_frame)

            resultMatrix_labels = []

            graph_frame = ctk.CTkFrame(master=self.result_frame)
            graph_frame.pack(side = "right", padx=5, pady=5, fill="both", expand=True)
            
            _empty = ctk.CTkLabel(graph_frame, text="No Graph", font = ctk.CTkFont(family="Arial", size = 16, weight = "bold"), text_color = "white")
            _empty.pack(padx = 5, pady = 5, fill = "both", expand = True)

            label_result = ctk.CTkLabel(result_matrix_frames[0],  # Frame for Hasil
                                    text=f"{self.matrixA.result:.2f}",
                                    font=ctk.CTkFont(family="Arial", size=16, weight="bold"),
                                    text_color="white",
                                    justify="center",
                                    anchor="center")
            label_result.pack(padx = 5, pady = 5, fill = "both", expand = True)

        elif mat == 'A':
            # result frame
            self.result_frame.pack(padx=5, pady=2, fill="both", expand=True)
            result_matrix_frames = []

            for i in range(out):
                # Result Labels Frame
                resultLabels_frame = ctk.CTkFrame(master=self.result_frame)
                resultLabels_frame.pack(side = "left", padx=5, pady=5, fill="both", expand=True)
                resultLabels_frame.columnconfigure(self.matrixA._getTuple('col'), weight=1, uniform="column")
                resultLabels_frame.rowconfigure(self.matrixA._getTuple('row'), weight=1, uniform="column")
                result_matrix_frames.append(resultLabels_frame)

            resultMatrix_labels = []

            graph_frame = ctk.CTkFrame(master=self.result_frame)
            graph_frame.pack(side = "right", padx=5, pady=5, fill="both", expand=True)
            
            _empty = ctk.CTkLabel(graph_frame, text="No Graph", font = ctk.CTkFont(family="Arial", size = 16, weight = "bold"), text_color = "white")
            _empty.pack(padx = 5, pady = 5, fill = "both", expand = True)

            # Buat entri baru sesuai row_count dan col_count matrix A
            for n in range(out):
                for i in range(self.row_countA):
                    row_labels = []
                    for j in range(self.matrixA.result.shape[1] if len(self.matrixA.result.shape) > 1 else 1):
                        label = ctk.CTkLabel(
                            result_matrix_frames[n],
                            text=f"{self.matrixA.result[i, j] if len(self.matrixA.result.shape) > 1 else self.matrixA.result[i]:.2f}",
                            width=50,
                            font=ctk.CTkFont(family="Arial", size=16, weight="bold"),
                            text_color="white",
                            justify="center",
                            anchor="center"
                        )
                        label.grid(row=i, column=j, padx=2, pady=2)
                        row_labels.append(label)
                    resultMatrix_labels.append(row_labels)

        elif mat == 'B' and L is not None and U is not None:
            # result frame
            self.result_frame.pack(padx=5, pady=2, fill="both", expand=True)
            result_matrix_frames = []

            for i in range(out):
                # Result Labels Frame
                resultLabels_frame = ctk.CTkFrame(master=self.result_frame)
                resultLabels_frame.pack(side = "left", padx=5, pady=5, fill="both", expand=True)
                resultLabels_frame.columnconfigure(self.matrixB._getTuple('col'), weight=1, uniform="column")
                resultLabels_frame.rowconfigure(self.matrixB._getTuple('row'), weight=1, uniform="column")
                result_matrix_frames.append(resultLabels_frame)

            resultMatrix_labels = []

            graph_frame = ctk.CTkFrame(master=self.result_frame)
            graph_frame.pack(side = "right", padx=5, pady=5, fill="both", expand=True)
            
            _empty = ctk.CTkLabel(graph_frame, text="No Graph", font = ctk.CTkFont(family="Arial", size = 16, weight = "bold"), text_color = "white")
            _empty.pack(padx = 5, pady = 5, fill = "both", expand = True)

            for i in range(self.row_countB):
                for j in range(self.col_countB):
                    # Result Labels for Matrix L
                    label = ctk.CTkLabel(result_matrix_frames[0],  # Frame for L
                                        text=f"{L[i, j]:.2f}",
                                        font=ctk.CTkFont(family="Arial", size=16, weight="bold"),
                                        text_color="white",
                                        justify="center",
                                        anchor="center")
                    label.grid(row=i, column=j, padx=2, pady=2)

            for i in range(self.row_countB):
                for j in range(self.col_countB):
                    # Result Labels for Matrix U
                    label_U = ctk.CTkLabel(result_matrix_frames[1],  # Frame for U
                                        text=f"{U[i, j]:.2f}",
                                        font=ctk.CTkFont(family="Arial", size=16, weight="bold"),
                                        text_color="white",
                                        justify="center",
                                        anchor="center")
                    label_U.grid(row=i, column=j, padx=2, pady=2)

            for i in range(self.row_countB):
                for j in range(self.col_countB):
                    # Result Labels untuk Matrix A
                    label_A = ctk.CTkLabel(result_matrix_frames[2],  # Frame for A
                                        text=f"{self.matrixB.matrix[i, j]:.2f}",
                                        font=ctk.CTkFont(family="Arial", size=16, weight="bold"),
                                        text_color="white",
                                        justify="center",
                                        anchor="center")
                    label_A.grid(row=i, column=j, padx=2, pady=2)
    
        elif mat == 'B' and op == 'cr':
            # result frame
            self.result_frame.pack(padx=5, pady=2, fill="both", expand=True)
            result_matrix_frames = []

            for i in range(out):
                # Result Labels Frame
                resultLabels_frame = ctk.CTkFrame(master=self.result_frame)
                resultLabels_frame.pack(side = "left", padx=5, pady=5, fill="both", expand=True)
                resultLabels_frame.columnconfigure(self.matrixB._getTuple('col'), weight=1, uniform="column")
                resultLabels_frame.rowconfigure(self.matrixB._getTuple('row'), weight=1, uniform="column")
                result_matrix_frames.append(resultLabels_frame)

            resultMatrix_labels = []

            graph_frame = ctk.CTkFrame(master=self.result_frame)
            graph_frame.pack(side = "right", padx=5, pady=5, fill="both", expand=True)
            
            _empty = ctk.CTkLabel(graph_frame, text="No Graph", font = ctk.CTkFont(family="Arial", size = 16, weight = "bold"), text_color = "white")
            _empty.pack(padx = 5, pady = 5, fill = "both", expand = True)

            for i in range(self.row_countB):
                for j in range(self.col_countB):
                    # Result Labels for Matrix A
                    label = ctk.CTkLabel(result_matrix_frames[0],  # Frame for A
                                        text=f"{self.matrixB.matrix[i, j]:.2f}",
                                        font=ctk.CTkFont(family="Arial", size=16, weight="bold"),
                                        text_color="white",
                                        justify="center",
                                        anchor="center")
                    label.grid(row=i, column=j, padx=2, pady=2)

            for i in range(self.row_countB):
                for j in range(0, 1):
                    # Result Labels for Matrix Variable
                    label_var = ctk.CTkLabel(result_matrix_frames[1],  # Frame for Variable
                                        text=f"{self.matrixB.result[i, j]:.2f}",
                                        font=ctk.CTkFont(family="Arial", size=16, weight="bold"),
                                        text_color="white",
                                        justify="center",
                                        anchor="center")
                    label_var.grid(row=i, column=j, padx=2, pady=2)

            for i in range(self.row_countB):
                for j in range(0, 1):
                    # Result Labels untuk Matrix C
                    label_C = ctk.CTkLabel(result_matrix_frames[2],  # Frame for C
                                        text=f"{self.matrixC.matrix[i, j]:.2f}",
                                        font=ctk.CTkFont(family="Arial", size=16, weight="bold"),
                                        text_color="white",
                                        justify="center",
                                        anchor="center")
                    label_C.grid(row=i, column=j, padx=2, pady=2)

        elif mat == 'B' and op == 'gj':
            # result frame
            self.result_frame.pack(padx=5, pady=2, fill="both", expand=True)
            result_matrix_frames = []

            for i in range(out):
                # Result Labels Frame
                resultLabels_frame = ctk.CTkFrame(master=self.result_frame)
                resultLabels_frame.pack(side = "left", padx=5, pady=5, fill="both", expand=True)
                resultLabels_frame.columnconfigure(self.matrixB._getTuple('col'), weight=1, uniform="column")
                resultLabels_frame.rowconfigure(self.matrixB._getTuple('row'), weight=1, uniform="column")
                result_matrix_frames.append(resultLabels_frame)

            resultMatrix_labels = []

            graph_frame = ctk.CTkFrame(master=self.result_frame)
            graph_frame.pack(side = "right", padx=5, pady=5, fill="both", expand=True)
            
            _empty = ctk.CTkLabel(graph_frame, text="No Graph", font = ctk.CTkFont(family="Arial", size = 16, weight = "bold"), text_color = "white")
            _empty.pack(padx = 5, pady = 5, fill = "both", expand = True)

            for i in range(self.row_countB):
                for j in range(self.col_countB):
                    # Result Labels for Matrix A
                    label = ctk.CTkLabel(result_matrix_frames[0],  # Frame for A
                                        text=f"{self.matrixB.matrix[i, j]:.2f}",
                                        font=ctk.CTkFont(family="Arial", size=16, weight="bold"),
                                        text_color="white",
                                        justify="center",
                                        anchor="center")
                    label.grid(row=i, column=j, padx=2, pady=2)

            for i in range(self.row_countB):
                for j in range(0, 1):
                    # Result Labels untuk Matrix Hasil
                    label_result = ctk.CTkLabel(result_matrix_frames[2],  # Frame for Hasil
                                        text=f"{self.matrixB.result[i, j]:.2f}",
                                        font=ctk.CTkFont(family="Arial", size=16, weight="bold"),
                                        text_color="white",
                                        justify="center",
                                        anchor="center")
                    label_result.grid(row=i, column=j, padx=2, pady=2)

            for i in range(self.row_countB):
                # Result Labels for Matrix Variable
                label_var = ctk.CTkLabel(result_matrix_frames[1],  # Frame for Variable
                                    text=f"{variable[i]:.2f}",
                                    font=ctk.CTkFont(family="Arial", size=16, weight="bold"),
                                    text_color="white",
                                    justify="center",
                                    anchor="center")
                label_var.grid(row=i, column=j, padx=2, pady=2)

        elif mat == 'B' and op == 'np':
            # header            
            result_header = ctk.CTkLabel(self.result_frame,
                                        text = "Result", 
                                        width = 50,
                                        font = ctk.CTkFont(family="Arial", size = 16, weight = "bold"),
                                        text_color = "white",
                                        justify = "left",
                                        anchor = "w"
                                        )
            result_header.pack(side = "top", padx=5, pady=5)

            # result frame
            self.result_frame.pack(padx=5, pady=2, fill="both", expand=True)
            result_matrix_frames = []

            for i in range(out):
                # Result Labels Frame
                resultLabels_frame = ctk.CTkFrame(master=self.result_frame)
                resultLabels_frame.pack(side = "left", padx=5, pady=5, fill="both", expand=True)
                resultLabels_frame.columnconfigure(self.matrixB._getTuple('col'), weight=1, uniform="column")
                resultLabels_frame.rowconfigure(self.matrixB._getTuple('row'), weight=1, uniform="column")
                result_matrix_frames.append(resultLabels_frame)

            resultMatrix_labels = []

            graph_frame = ctk.CTkFrame(master=self.result_frame)
            graph_frame.pack(side = "right", padx=5, pady=5, fill="both", expand=True)
           
            # Buat plot grafik
            self.plot_frame = ctk.CTkFrame(graph_frame)
            self.plot_frame.pack(fill="both", expand=True, padx=10, pady=10)

            # Create a plot
            self.create_plot("B")

            for i in range(self.matrixB.matrix.shape[0] + 1):
                for j in range(self.col_countB):
                    # Result Labels for Matrix A
                    label = ctk.CTkLabel(result_matrix_frames[0],  # Frame for A
                                        text=f"{self.matrixB.matrix[i, j]:.2f}",
                                        font=ctk.CTkFont(family="Arial", size=16, weight="bold"),
                                        text_color="white",
                                        justify="center",
                                        anchor="center")
                    label.grid(row=i, column=j, padx=2, pady=2)

        elif mat == 'B' and op == 'jac':
            # result frame
            self.result_frame.pack(padx=5, pady=2, fill="both", expand=True)
            result_matrix_frames = []

            for i in range(out):
                # Result Labels Frame
                resultLabels_frame = ctk.CTkFrame(master=self.result_frame)
                resultLabels_frame.pack(side = "left", padx=5, pady=5, fill="both", expand=True)
                resultLabels_frame.columnconfigure(self.matrixA._getTuple('col'), weight=1, uniform="column")
                resultLabels_frame.rowconfigure(self.matrixA._getTuple('row'), weight=1, uniform="column")
                result_matrix_frames.append(resultLabels_frame)

            resultMatrix_labels = []

            graph_frame = ctk.CTkFrame(master=self.result_frame)
            graph_frame.pack(side = "right", padx=5, pady=5, fill="both", expand=True)
            
            _empty = ctk.CTkLabel(graph_frame, text="No Graph", font = ctk.CTkFont(family="Arial", size = 16, weight = "bold"), text_color = "white")
            _empty.pack(padx = 5, pady = 5, fill = "both", expand = True)

            for i in range(len(self.matrixB.result)):
                label_result = ctk.CTkLabel(result_matrix_frames[0],  # Frame for Hasil
                                        text=f"{self.matrixB.result[i]}",
                                        font=ctk.CTkFont(family="Arial", size=16, weight="bold"),
                                        text_color="white",
                                        justify="center",
                                        anchor="center"
                                        )
                label_result.pack(side = "left", padx = 5, pady = 5, fill = "both", expand = True)
                resultMatrix_labels.append(label_result)

        elif mat == 'B' and op == 'det':
            # result frame
            self.result_frame.pack(padx=5, pady=2, fill="both", expand=True)
            result_matrix_frames = []

            for i in range(out):
                # Result Labels Frame
                resultLabels_frame = ctk.CTkFrame(master=self.result_frame)
                resultLabels_frame.pack(side = "left", padx=5, pady=5, fill="both", expand=True)
                resultLabels_frame.columnconfigure(self.matrixA._getTuple('col'), weight=1, uniform="column")
                resultLabels_frame.rowconfigure(self.matrixA._getTuple('row'), weight=1, uniform="column")
                result_matrix_frames.append(resultLabels_frame)

            resultMatrix_labels = []

            graph_frame = ctk.CTkFrame(master=self.result_frame)
            graph_frame.pack(side = "right", padx=5, pady=5, fill="both", expand=True)
            
            _empty = ctk.CTkLabel(graph_frame, text="No Graph", font = ctk.CTkFont(family="Arial", size = 16, weight = "bold"), text_color = "white")
            _empty.pack(padx = 5, pady = 5, fill = "both", expand = True)

            label_result = ctk.CTkLabel(result_matrix_frames[0],  # Frame for Hasil
                                    text=f"{self.matrixB.result:.2f}",
                                    font=ctk.CTkFont(family="Arial", size=16, weight="bold"),
                                    text_color="white",
                                    justify="center",
                                    anchor="center")
            label_result.pack(padx = 5, pady = 5, fill = "both", expand = True)

        elif mat == 'B':
            # result frame
            self.result_frame.pack(padx=5, pady=2, fill="both", expand=True)
            result_matrix_frames = []

            for i in range(out):
                # Result Labels Frame
                resultLabels_frame = ctk.CTkFrame(master=self.result_frame)
                resultLabels_frame.pack(side = "left", padx=5, pady=5, fill="both", expand=True)
                resultLabels_frame.columnconfigure(self.matrixB._getTuple('col'), weight=1, uniform="column")
                resultLabels_frame.rowconfigure(self.matrixB._getTuple('row'), weight=1, uniform="column")
                result_matrix_frames.append(resultLabels_frame)

            resultMatrix_labels = []

            graph_frame = ctk.CTkFrame(master=self.result_frame)
            graph_frame.pack(side = "right", padx=5, pady=5, fill="both", expand=True)
            
            _empty = ctk.CTkLabel(graph_frame, text="No Graph", font = ctk.CTkFont(family="Arial", size = 16, weight = "bold"), text_color = "white")
            _empty.pack(padx = 5, pady = 5, fill = "both", expand = True)

            # Buat entri baru sesuai row_countB dan col_count matrix A
            for n in range(out):
                for i in range(self.row_countB):
                    row_labels = []
                    for j in range(self.matrixB.result.shape[1] if len(self.matrixB.result.shape) > 1 else 1):
                        label = ctk.CTkLabel(
                            result_matrix_frames[n],
                            text=f"{self.matrixB.result[i, j] if len(self.matrixB.result.shape) > 1 else self.matrixB.result[i]:.2f}",
                            width=50,
                            font=ctk.CTkFont(family="Arial", size=16, weight="bold"),
                            text_color="white",
                            justify="center",
                            anchor="center"
                        )
                        label.grid(row=i, column=j, padx=2, pady=2)
                        row_labels.append(label)
                    resultMatrix_labels.append(row_labels)
        
    def create_plot(self, mat):
        if mat == "A":
            # Gambar canvas matplotliv
            fig = Figure(figsize=(6, 4), dpi=100, facecolor='white', edgecolor='black')
            ax = fig.add_subplot(111)
            ax.plot(self.matrixA._xdata, self.matrixA._ydata, marker='o', linestyle='-', color='blue', label='Data Line')
            ax.set_title("Data Plot")
            ax.set_xlabel("X-axis")
            ax.set_ylabel("Y-axis")
            ax.legend()
            ax.grid(True)

            # Embed the Matplotlib figure into the customtkinter frame
            canvas = FigureCanvasTkAgg(fig, master=self.plot_frame)
            canvas_widget = canvas.get_tk_widget()
            canvas_widget.pack(fill="both", expand=True)
            canvas.draw()

        if mat == "B":
            # Gambar canvas matplotliv
            fig = Figure(figsize=(6, 4), dpi=100, facecolor='white', edgecolor='black')
            ax = fig.add_subplot(111)
            ax.plot(self.matrixB._xdata, self.matrixB._ydata, marker='o', linestyle='-', color='blue', label='Data Line')
            ax.set_title("Data Plot")
            ax.set_xlabel("X-axis")
            ax.set_ylabel("Y-axis")
            ax.legend()
            ax.grid(True)

            # Embed the Matplotlib figure into the customtkinter frame
            canvas = FigureCanvasTkAgg(fig, master=self.plot_frame)
            canvas_widget = canvas.get_tk_widget()
            canvas_widget.pack(fill="both", expand=True)
            canvas.draw()


if __name__ == "__main__":
    app = App()
    app.root.mainloop()