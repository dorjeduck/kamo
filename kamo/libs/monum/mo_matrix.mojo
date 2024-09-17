from algorithm import vectorize,parallelize
from algorithm.functional import elementwise
from memory.memory import memcpy,memset_zero
from python import Python,PythonObject
from random import rand

from .mo_num import add, sub, mul, div


struct MoMatrix[dtype: DType, simd_width: Int](
    Stringable, CollectionElement, Sized
):
    var _mat_ptr: UnsafePointer[Scalar[dtype]]
    var rows: Int
    var cols: Int
   
    @always_inline
    fn __init__(inout self, rows: Int, cols: Int):
        self._mat_ptr = UnsafePointer[Scalar[dtype]].alloc(rows * cols)

        self.rows = rows
        self.cols = cols
        
        memset_zero(self._mat_ptr, self.rows * self.cols)

    fn __init__(inout self, rows: Int, cols: Int, val: Scalar[dtype]):
        self._mat_ptr = UnsafePointer[Scalar[dtype]].alloc(rows * cols)
        
        self.rows = rows
        self.cols = cols
       
        @parameter
        fn _set_val[width: Int](iv: Int) -> None:
            self._mat_ptr.store[width=width](iv, val)

        vectorize[_set_val, simd_width](self.rows * self.cols)

    @always_inline
    fn __init__(inout self, rows: Int, cols: Int, *data: Scalar[dtype]):
        var data_len = len(data)
        self.rows = rows
        self.cols = cols
      
        self._mat_ptr = UnsafePointer[Scalar[dtype]].alloc(data_len)
        
        for i in range(data_len):
            self._mat_ptr[i] = data[i]

    @always_inline
    fn __init__(
        inout self, rows: Int, cols: Int, owned list: List[Scalar[dtype]]
    ):
        var list_len = len(list)
        self.rows = rows
        self.cols = cols
       
        self._mat_ptr = UnsafePointer[Scalar[dtype]].alloc(list_len)
       
        for i in range(list_len):
            self._mat_ptr[i] = list[i]

    @always_inline
    fn __init__(
        inout self, rows: Int, cols: Int,  mv: MoVector[dtype,simd_width]
    ):
        
        self.rows = rows
        self.cols = cols
        self._mat_ptr = UnsafePointer[Scalar[dtype]].alloc(self.rows*self.cols)
       
        for i in range(self.rows*self.cols):
            self._mat_ptr[i] = mv[i]

    @always_inline
    fn __init__(
        inout self, rows: Int, cols: Int, owned ptr: UnsafePointer[Scalar[dtype]]
    ):
        self.rows = rows
        self.cols = cols
        self._mat_ptr = ptr

    fn __del__(owned self):
        self._mat_ptr.free()

    @always_inline
    fn __copyinit__(inout self, other: Self):
        # print("cop")
        self.rows = other.rows
        self.cols = other.cols
        self._mat_ptr = UnsafePointer[Scalar[dtype]].alloc(self.rows * self.cols)
        # print("mm alloc", self._mat_ptr)
        memcpy(self._mat_ptr, other._mat_ptr, self.rows * self.cols)

    fn __moveinit__(inout self, owned existing: Self):
        # print("mov")
        self._mat_ptr = existing._mat_ptr
        self.rows = existing.rows
        self.cols = existing.cols
        existing.rows = 0
        existing.cols = 0
        existing._mat_ptr = UnsafePointer[Scalar[dtype]]()

    fn shape(self) -> String:
        return  "("+str(self.rows)+","+str(self.cols)+")"

    @always_inline
    fn transpose(self) -> MoMatrix[dtype, simd_width]:
        return MoMatrix[dtype, simd_width](
            self.cols, self.rows, self._transpose_order()
        )

    fn zero(inout self):
        memset_zero(self._mat_ptr, self.rows * self.cols)

    fn __setitem__(inout self, elem: Int, val: Scalar[dtype]):
        self._mat_ptr.store[width=1](elem, val)

    fn __setitem__(inout self, row: Int, col: Int, val: SIMD[dtype, 1]):
        return self._mat_ptr.store[width=1](row * self.cols + col, val)

    @always_inline
    fn __getitem__(self, idx: Int) -> SIMD[dtype, 1]:
        return self._mat_ptr.load(idx)

    @always_inline
    fn __getitem__(self, x: Int, y: Int) -> SIMD[dtype, 1]:
        return self._mat_ptr.load(x * self.cols + y)

    @always_inline
    fn __len__(self) -> Int:
        return self.rows * self.cols

    fn load[width: Int](self, x: Int, y: Int) -> SIMD[dtype, width]:
        return self._mat_ptr.load[width=width](x * self.cols + y)

    fn store[width: Int](self, x: Int, y: Int, val: SIMD[dtype, width]):
        return self._mat_ptr.store[width=width](x * self.cols + y, val)

    '''
    fn __matmul__(self, mat: Self) -> MoMatrix[dtype, simd_width]:
        var res = MoMatrix[dtype, simd_width](self.rows, mat.cols)
        
        @parameter
        fn calc_row(m: Int):
            for k in range(self.cols):
                @parameter
                fn dot[width: Int](iv: Int):
                    res.store[width](
                        m,
                        iv,
                        res.load[width](m, iv)
                        + self[m, k] * mat.load[width](k, iv),
                    )

                vectorize[dot, simd_width](size=res.cols)
        parallelize[calc_row](res.rows, res.rows)

        return res
    
    '''
    fn __matmul__(self, mat: Self) -> MoMatrix[dtype, simd_width]:
        var res = MoMatrix[dtype, simd_width](self.rows, mat.cols)
        ##print(self.shape(),mat.shape())
        for m in range(res.rows):
            for k in range(self.cols):
                @parameter
                fn dot[width: Int](iv: Int):
                    res.store[width](
                        m,
                        iv,
                        res.load[width](m, iv)
                        + self[m, k] * mat.load[width](k, iv),
                    )

                vectorize[dot, simd_width](size=res.cols)
                
        return res
    
    fn __matmul__(
        self, vec: MoVector[dtype, simd_width]
    ) -> MoVector[dtype, simd_width]:
        if self.cols != vec.size:
            print(
                "trouble __matmul__",
                "vec.size",
                vec.size,
                "self.cols",
                self.cols,
                "self.rows",
                self.rows,
            )

        var res = MoVector[dtype, simd_width](self.rows)
        for m in range(self.rows):

            @parameter
            fn dot[width: Int](iv: Int):
                res.store[width](
                    m,
                    res.load[width](m)
                    + self.load[width](m, iv) * vec.load[width](iv),
                )

            vectorize[dot, simd_width](size=self.cols)

        return res

    fn __rmatmul__(
        self, vec: MoVector[dtype, simd_width]
    ) -> MoVector[dtype, simd_width]:
        if vec.size != self.rows:
            print("trouble __rmatmul__")
        return self.transpose().__matmul__(vec)

    @always_inline
    fn __mul__(self, mat: Self) -> Self:
       
        return MoNum[dtype, simd_width]._elemwise_matrix_matrix[mul](self, mat)

    @always_inline
    fn __mul__(self, val: Scalar[dtype]) -> Self:
        return MoNum[dtype, simd_width]._elemwise_scalar_math[mul](self, val)

    @always_inline
    fn __rmul__(self, val: Scalar[dtype]) -> Self:
        return self.__mul__(val)

    @always_inline
    fn __add__(self, mat: Self) -> Self:
        return MoNum[dtype, simd_width]._elemwise_matrix_matrix[add](self, mat)

    @always_inline
    fn __add__(self, val: Scalar[dtype]) -> Self:
        return MoNum[dtype, simd_width]._elemwise_scalar_math[add](self, val)

    @always_inline
    fn __sub__(self, mat: Self) -> Self:
        return MoNum[dtype, simd_width]._elemwise_matrix_matrix[sub](self, mat)

    @always_inline
    fn __sub__(self, val: Scalar[dtype]) -> Self:
        return MoNum[dtype, simd_width]._elemwise_scalar_math[sub](self, val)


    @always_inline
    fn __iadd__(inout self, other: Self):
        @parameter
        fn matrix_matrix_vectorize[width: Int](iv: Int) -> None:
            self._mat_ptr.store[width=width](
                iv,
                self._mat_ptr.load[width=width](iv)
                + other._mat_ptr.load[width=width](iv),
            )

        vectorize[matrix_matrix_vectorize, simd_width](len(self))

    @always_inline
    fn __isub__(inout self, other: Self):
        self.__iadd__(-other)

    @always_inline
    fn __truediv__(self, s: Scalar[dtype]) -> Self:
        return MoNum[dtype, simd_width]._elemwise_scalar_math[div](self, s)

    @always_inline
    fn __truediv__(self, mat: Self) -> Self:
        return MoNum[dtype, simd_width]._elemwise_matrix_matrix[div](self, mat)

    @always_inline
    fn __pow__(self, p: Int) -> Self:
        return MoNum[dtype, simd_width]._elemwise_func_pow(self, p)

    @always_inline
    fn __eq__(self, val: Int) -> Self:
        var bool_float_array = Self(self.rows, self.cols)

        @parameter
        fn elemwise_vectorize[width: Int](iv: Int) -> None:
            var iters = SIMD[dtype, width](0)
            bool_float_array._mat_ptr.store[width=width](
                iv,
                (self._mat_ptr.load[width=width](iv) == val).select(
                    iters + 1, iters
                ),
            )

        vectorize[elemwise_vectorize, self.simd_width](len(self))
        return bool_float_array

    @always_inline
    fn __neg__(self) -> Self:
        return MoNum[dtype, simd_width]._elemwise_scalar_math[mul](self, -1)

    @always_inline
    fn __itruediv__(inout self, s: Scalar[dtype]):
        @parameter
        fn elemwise_vectorize[width: Int](iv: Int) -> None:
            self._mat_ptr.store[width=width](
                iv,
                div[dtype, width](
                    self._mat_ptr.load[width=width](iv), SIMD[dtype, width](s)
                ),
            )

        vectorize[elemwise_vectorize, simd_width](len(self))

    fn __str__(self) -> String:
        var printStr: String = "["

        for i in range(self.rows):
            for j in range(self.cols):
                if j == 0:
                    printStr += "["
                printStr += str(self[i, j])
                if j < self.cols - 1:
                    printStr += ", "
                elif i < self.rows - 1:
                    printStr += "]\n"
                else:
                    printStr += "]]\n"
        printStr += (
            "MoMatrix: "
            + str(self.rows)
            + "x"
            + str(self.cols)
            + " | "
            + "DType:"
            + str(dtype)
            + "\n"
        )
        return printStr

    @always_inline
    fn formated_str(self, digits: Int = 2) -> String:
        var printStr: String = "["

        for i in range(self.rows):
            for j in range(self.cols):
                if j == 0:
                    printStr += "["
                printStr += format_float(self[i, j], digits)
                if j < self.cols - 1:
                    printStr += ", "
                elif i < self.rows - 1:
                    printStr += "]\n"
                else:
                    printStr += "]]\n"
        printStr += (
            "MoMatrix: "
            + str(self.rows)
            + "x"
            + str(self.cols)
            + " | "
            + "DType:"
            + str(dtype)
            + "\n"
        )
        return printStr

    @always_inline
    fn insert(self, mv: MoVector[dtype, simd_width], start_pos: Int):
        # for i in range(len(v)):
        #    self._mat_ptr[pos+i]=v._vec_ptr[i]

        memcpy(self._mat_ptr.offset(start_pos), mv._vec_ptr, len(mv))

    @always_inline
    fn insert(self, val: Scalar[dtype], start_pos: Int, num: Int, step: Int = 1):
        ##todo optimize?
        #for i in range(num):
        #    self._mat_ptr[start_pos + i * step] = val

        @parameter
        fn _val[width:Int](iv:Int):
            self._mat_ptr.offset(start_pos+iv*step).strided_store[width=width](val,step)
        
        
        vectorize[_val,simd_width](size=num)


    @always_inline
    fn insert(self, mv: MoVector[dtype, simd_width], start_pos: Int, step: Int):
        #todo optimize?
        #for i in range(len(mv)):
        #    self._mat_ptr[start_pos + i * step] = mv[i]    

        @parameter
        fn _val[width:Int](iv:Int):
            self._mat_ptr.offset(start_pos+iv*step).strided_store[width=width](mv.load[width](iv),step)
        vectorize[_val,simd_width](size=mv.size)

    @always_inline
    fn insert_row(self, row: Int, val: Scalar[dtype]):
        self.insert(val, row*self.cols, self.cols,1)

    @always_inline
    fn insert_row(self, row: Int, mv: MoVector[dtype, simd_width]):
        memcpy(self._mat_ptr.offset(row * self.cols), mv._vec_ptr, len(mv))

    @always_inline
    fn insert_col(self, col: Int, val: Scalar[dtype]):
        self.insert(val, col, self.rows, self.cols)

    @always_inline
    fn insert_col(self, col: Int, mv: MoVector[dtype, simd_width]):
        self.insert(mv, col, self.cols)

    @always_inline
    fn get_row(self, row: Int) -> MoVector[dtype, simd_width]:
        var res = MoVector[dtype, simd_width](self.cols)
        self.get_row(row, res)
        return res

    @always_inline
    fn get_row(self, row: Int, inout mv: MoVector[dtype, simd_width]):
        # @parameter
        # fn _getrow[width:Int](iv:Int):
        #    mv.store[width](iv,self.load[width](row,iv))
        # vectorize[_getrow,simd_width](self.cols)

        memcpy(mv._vec_ptr, self._mat_ptr.offset(row * self.cols), len(mv))

    @always_inline
    fn get_col(self, col: Int) -> MoVector[dtype, simd_width]:
        var res = MoVector[dtype, simd_width](self.rows)

        #for i in range(self.rows):
        #    res[i] = self[i, col]
              
        @parameter
        fn _get_col[width:Int](iv:Int):
             res._vec_ptr.store[width=width](iv,self._mat_ptr.offset(col+iv*self.cols).strided_load[width=width](self.cols))
        

        vectorize[_get_col,simd_width](self.rows)
        
        return res

    @always_inline
    fn flatten(self)->MoVector[dtype,simd_width]:
        var res = MoVector[dtype, simd_width](self.rows*self.cols)

        memcpy(res._vec_ptr,self._mat_ptr,self.rows*self.cols)

        return res


    @staticmethod
    fn rand(
        rows: Int,
        cols: Int,
        low: Scalar[dtype] = 0.0,
        high: Scalar[dtype] = 1.0,
    ) -> Self:
        var res = MoMatrix[dtype, simd_width](rows, cols)
        rand(res._mat_ptr, rows * cols)

        if low != 0 or high != 1:

            @parameter
            fn _map[width: Int](iv: Int):
                res._mat_ptr.store[width=width](
                    iv,
                    (low + res._mat_ptr.load[width=width](iv) * (high - low)),
                )

            vectorize[_map, simd_width](rows * cols)

        return res

    @staticmethod
    fn diag(vec: MoVector[dtype]) -> Self:
        var res = Self(vec.size, vec.size)
        for i in range(vec.size):
            res._mat_ptr.store[width=1](i * vec.size + i, vec[i])
        return res

    @staticmethod
    fn from_numpy(np_array: PythonObject) raises -> Self:
        var np_array_ptr = UnsafePointer[Scalar[dtype]](
            __mlir_op.`pop.index_to_pointer`[
                _type = __mlir_type[`!kgen.pointer<scalar<`, dtype.value, `>>`]
            ](
                SIMD[DType.index, 1](
                    np_array.__array_interface__["data"][0].__index__()
                ).value
            )
        )
        var rows = int(np_array.shape[0])
        var cols = int(np_array.shape[1])

        var out = MoMatrix[dtype, simd_width](rows, cols)
        memcpy(out._mat_ptr, np_array_ptr, rows * cols)

        return out

    fn to_numpy(self) raises -> PythonObject:
        var np = Python.import_module("numpy")

        var type = np.float32
        if dtype == DType.float64:
            type = np.float64
        elif dtype == DType.float16:
            type = np.float16

        var np_arr = np.zeros((self.rows, self.cols), dtype=type)

        '''
        var np_array_ptr = UnsafePointer[Scalar[dtype]](
            __mlir_op.`pop.index_to_pointer`[
                _type = __mlir_type[`!kgen.pointer<scalar<`, dtype.value, `>>`]
            ](
                SIMD[DType.index, 1](
                    np_arr.__array_interface__["data"][0].__index__()
                ).value
            )
        )
        '''

        memcpy(np_arr.__array_interface__['data'][0].unsafe_get_as_pointer[dtype](), self._mat_ptr, self.rows * self.cols)
       

        #memcpy(np_array_ptr, self._mat_ptr, self.rows * self.cols)
        
        return np_arr^

    fn _transpose_order(self) -> UnsafePointer[Scalar[dtype]]:
        var new_ptr = UnsafePointer[Scalar[dtype]].alloc(self.rows * self.cols)

        for idx_col in range(self.cols):
            var tmp_ptr = self._mat_ptr.offset(idx_col)

            @parameter
            fn convert[width: Int](iv: Int) -> None:
                new_ptr.store[width=width](
                    iv + idx_col * self.rows,
                    tmp_ptr.strided_load[width=width](self.cols),
                )
                tmp_ptr = tmp_ptr.offset(width * self.cols)

            vectorize[convert, simd_width](self.rows)
        return new_ptr
