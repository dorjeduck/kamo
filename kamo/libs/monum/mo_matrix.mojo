from algorithm import vectorize
from algorithm.functional import elementwise
from math import mul, div, pow,sin,exp,mod, add, trunc, align_down, align_down_residual
from python import Python
from random import rand


struct MoMatrix[
    dtype: DType = DType.float32, simd_width: Int = 2 * simdwidthof[dtype]()
](Stringable, CollectionElement, Sized):
    var _mat_ptr: DTypePointer[dtype]
    var rows: Int
    var cols: Int

    @always_inline
    fn __init__(inout self, rows: Int, cols: Int):
        self._mat_ptr = DTypePointer[dtype].alloc(rows * cols)
        self.rows = rows
        self.cols = cols
        memset_zero[dtype](self._mat_ptr, self.rows * self.cols)

    fn __init__(inout self, rows: Int, cols: Int, val: Scalar[dtype]):
        self._mat_ptr = DTypePointer[dtype].alloc(rows * cols)
        self.rows = rows
        self.cols = cols

        @parameter
        fn splat_val[nelts: Int](idx: Int) -> None:
            self._mat_ptr.store[width=nelts](
                idx, self._mat_ptr.load[width=nelts](idx).splat(val)
            )

        vectorize[splat_val, simd_width](self.rows * self.cols)

    @always_inline
    fn __init__(inout self, rows: Int, cols: Int, *data: Scalar[dtype]):
        var data_len = len(data)
        self.rows = rows
        self.cols = cols
        self._mat_ptr = DTypePointer[dtype].alloc(data_len)
        for i in range(data_len):
            self._mat_ptr[i] = data[i]

    @always_inline
    fn __init__(
        inout self, rows: Int, cols: Int, owned list: List[Scalar[dtype]]
    ):
        var list_len = len(list)
        self.rows = rows
        self.cols = cols
        self._mat_ptr = DTypePointer[dtype].alloc(list_len)
        for i in range(list_len):
            self._mat_ptr[i] = list[i]

    @always_inline
    fn __init__(inout self, rows: Int, cols: Int, ptr: DTypePointer[dtype]):
        self.rows = rows
        self.cols = cols
        self._mat_ptr = DTypePointer[dtype].alloc(self.rows * self.cols)
        for i in range(self.rows * self.cols):
            self._mat_ptr[i] = ptr[i]

    @always_inline
    fn transpose(self) -> MoMatrix[dtype,simd_width]:
        return MoMatrix[dtype,simd_width](self.cols, self.rows, self._transpose_order())

    fn __del__(owned self):
        self._mat_ptr.free()

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
    fn __copyinit__(inout self, other: Self):
        self.rows = other.rows
        self.cols = other.cols
        self._mat_ptr = DTypePointer[dtype].alloc(self.rows * self.cols)
        memcpy(self._mat_ptr, other._mat_ptr, self.rows * self.cols)

    fn __moveinit__(inout self, owned existing: Self):
        self._mat_ptr = existing._mat_ptr
        self.rows = existing.rows
        self.cols = existing.cols
        existing.rows = 0
        existing.cols = 0
        existing._mat_ptr = DTypePointer[dtype]()

    @always_inline
    fn __len__(self) -> Int:
        return self.rows * self.cols

    fn load[nelts: Int](self, x: Int, y: Int) -> SIMD[dtype, nelts]:
        return self._mat_ptr.load[width=nelts](x * self.cols + y)

    fn store[nelts: Int](self, x: Int, y: Int, val: SIMD[dtype, nelts]):
        return self._mat_ptr.store[width=nelts](x * self.cols + y, val)

    fn __matmul__(self, mat: Self) -> MoMatrix[dtype,simd_width]:
        var res = MoMatrix[dtype,simd_width](self.rows, mat.cols)

        for m in range(res.rows):
            for k in range(self.cols):

                @parameter
                fn dot[nelts: Int](iv: Int):
                    res.store[nelts](
                        m,
                        iv,
                        res.load[nelts](m, iv)
                        + self[m, k] * mat.load[nelts](k, iv),
                    )

                vectorize[dot, simd_width](size=res.cols)

        return res

    fn __matmul__(self, vec: MoVector[dtype,simd_width]) -> MoVector[dtype,simd_width]:
        
        if self.cols != vec.size:
            print("trouble __matmul__","vec.size",vec.size,"self.cols",self.cols,"self.rows",self.rows)
        
        var res = MoVector[dtype,simd_width](self.rows)
        for m in range(self.rows):
            @parameter
            fn dot[nelts: Int](iv: Int):
                 res.store[nelts](
                    m,
                    res.load[nelts](m) + 
                    self.load[nelts](m, iv) * vec.load[nelts](iv) 
                )
            vectorize[dot, simd_width](size=self.cols)

        return res

    fn __rmatmul__(self, vec: MoVector[dtype,simd_width]) -> MoVector[dtype,simd_width]:
        
        if vec.size != self.rows:
            print("trouble __rmatmul__")
        return self.transpose().__matmul__(vec)
   

    @always_inline
    fn __mul__(self,  mat: Self) -> Self:
        return MoNum[dtype,simd_width]._elemwise_matrix_matrix[mul](self,mat)

    # @always_inline
    # fn __mul__(self, vec: MoVector)->Self:
    #    return self._el[mul](vec)

    @always_inline
    fn __mul__(self, val: Scalar[dtype]) -> Self:
        return MoNum[dtype,simd_width]._elemwise_scalar_math[mul](self,val)
       

    @always_inline
    fn __rmul__(self, val: Scalar[dtype]) -> Self:
        return self.__mul__(val)

    @always_inline
    fn __add__(self, mat: Self) -> Self:
        return MoNum[dtype,simd_width]._elemwise_matrix_matrix[add](self,mat)

    @always_inline
    fn __add__(self, val: Scalar[dtype]) -> Self:
        return MoNum[dtype,simd_width]._elemwise_scalar_math[math.add](self,val)
       

    @always_inline
    fn __iadd__(inout self, other: Self):

        @parameter
        fn matrix_matrix_vectorize[nelts: Int](iv: Int) -> None:
            self._mat_ptr.store[width=nelts](
                iv,
                self._mat_ptr.load[width=nelts](iv)
                + other._mat_ptr.load[width=nelts](iv),
            )

        vectorize[matrix_matrix_vectorize, simd_width](len(self))

    @always_inline
    fn __truediv__( self, s: Scalar[dtype]) -> Self:
        return MoNum[dtype,simd_width]._elemwise_scalar_math[math.div](self,s)
        

    @always_inline
    fn __truediv__( self, mat: Self) -> Self:
        return MoNum[dtype,simd_width]._elemwise_matrix_matrix[div](self,mat)

    @always_inline
    fn __pow__( self, p: Int) -> Self:
        return MoNum[dtype,simd_width]._elemwise_scalar_math[math.pow](self,p)

    @always_inline
    fn __eq__( self, val: Int) -> Self:
        
        var bool_float_array = Self(self.rows,self.cols)

        @parameter
        fn elemwise_vectorize[nelts: Int](iv: Int) -> None:
            var iters = SIMD[dtype, nelts](0)
            bool_float_array._mat_ptr.store[width=nelts](
                iv,
                (self._mat_ptr.load[width=nelts](iv) == val).select(
                    iters + 1, iters
                ),
            )

        vectorize[elemwise_vectorize, self.simd_width](len(self))
        return bool_float_array

    @always_inline
    fn __neg__( self) -> Self:
        return MoNum[dtype,simd_width]._elemwise_scalar_math[mul](self,-1)
    
    @always_inline
    fn __itruediv__(inout self, s: Scalar[dtype]):
      
        @parameter
        fn elemwise_vectorize[nelts: Int](iv: Int) -> None:
            self._mat_ptr.store[width=nelts](
                iv,
                math.div[dtype, nelts](
                    self._mat_ptr.load[width=nelts](iv), SIMD[dtype, nelts](s)
                ),
            )

        vectorize[elemwise_vectorize, simd_width](len(self))

   

    @always_inline
    fn sum( self, axis: Int) -> MoVector[dtype]:
        if axis == 0:
            var res = MoVector[dtype](self.cols)
            for j in range(self.cols):
                for i in range(self.rows):
                    res[j] += self[i, j]
            return res
        else:
            var res = MoVector[dtype](self.rows)
            for i in range(self.rows):

                @parameter
                fn _sum_row[nelts: Int](iv: Int):
                    res[i] += self.load[nelts](i, iv).reduce_add()

                vectorize[_sum_row, simd_width](self.cols)
            return res

    @always_inline
    fn flatten(self) -> MoVector[dtype,simd_width]:
        return MoVector[dtype,simd_width](self.rows*self.cols, self._mat_ptr)

    fn __str__(self) -> String:
       
        var prec: Int = 4
        var printStr: String = "["
        
        for i in range(self.rows):
            for j in range(self.cols):
                if j == 0:
                    printStr += "["
                printStr += self[i,j]
                if j <self.cols-1:
                    printStr += ", "
                elif i<self.rows-1:
                    printStr +="]\n"
                else:
                    printStr +="]]\n"
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
    fn insert(self,mv:MoVector[dtype,simd_width],start_pos:Int):
        
        #for i in range(len(v)):
        #    self._mat_ptr[pos+i]=v._vec_ptr[i]
        
        memcpy(self._mat_ptr.offset(start_pos), mv._vec_ptr, len(mv))
    
    @always_inline
    fn insert(self,val:Scalar[dtype],start_pos:Int,num:Int,step:Int=1):
        ##todo optimize?
        for i in range(num):
            self._mat_ptr[start_pos+i*step]=val

    @always_inline
    fn insert(self,mv:MoVector[dtype,simd_width],start_pos:Int,step:Int=1):
        ##todo optimize?
        for i in range(len(mv)):
            self._mat_ptr[start_pos+i*step]=mv[i]

    @always_inline
    fn col_insert(self,val:Scalar[dtype],col:Int):
        self.insert(val,col,self.rows,self.cols)
    
    @always_inline
    fn col_insert(self,mv:MoVector[dtype,simd_width],col:Int):
        self.insert(mv,col,self.rows)

    
    @always_inline
    fn get_row(self,row:Int) -> MoVector[dtype,simd_width]:
        ##todo optimize
        var res = MoVector[dtype,simd_width](self.rows)

        for i in range(self.cols):
            res[i] = self[row,i]
            
        return res
    
    @always_inline
    fn get_col(self,col:Int) -> MoVector[dtype,simd_width]:
        var res = MoVector[dtype,simd_width](self.rows)

        for i in range(self.rows):
            res[i] = self[i,col]

        return res
        
    
    @staticmethod
    fn rand(*dims: Int) -> Self:
        var _mat_ptr = DTypePointer[dtype].alloc(dims[0] * dims[1])
        rand(_mat_ptr, dims[0] * dims[1])
        return Self(dims[0], dims[1], _mat_ptr)

    @staticmethod
    fn diag(vec: MoVector[dtype]) -> Self:
        var res = Self(vec.size, vec.size)
        for i in range(vec.size):
            res._mat_ptr.store[width=1](i * vec.size + i, vec[i])
        return res
    
    @staticmethod
    fn from_numpy(np_array: PythonObject) raises -> Self:
        var np_array_ptr = DTypePointer[dtype](
        __mlir_op.`pop.index_to_pointer`[
            _type = __mlir_type[`!kgen.pointer<scalar<`, dtype.value, `>>`]
        ](
            SIMD[DType.index,1](np_array.__array_interface__['data'][0].__index__()).value
        )
    )
        var rows = int(np_array.shape[0])
        var cols = int(np_array.shape[1])
        var _mat_ptr = DTypePointer[dtype].alloc(rows*cols)
        memcpy(_mat_ptr, np_array_ptr, rows*cols)
        var out = Self(rows,cols,_mat_ptr)
        return out ^

    fn to_numpy(self) raises -> PythonObject:
        var np = Python.import_module("numpy")

        var type = np.float32
        if dtype == DType.float64:
            type =np.float64
        elif dtype == DType.float16:
            type =np.float16

        var np_arr = np.zeros((self.rows,self.cols), dtype=type)
         
        var np_array_ptr = DTypePointer[dtype](
        __mlir_op.`pop.index_to_pointer`[
            _type = __mlir_type[`!kgen.pointer<scalar<`, dtype.value, `>>`]
        ](
            SIMD[DType.index,1](np_arr.__array_interface__['data'][0].__index__()).value
        )
        )

        if dtype == DType.float32:
            memcpy(np_array_ptr, self._mat_ptr, self.rows*self.cols)
        else:
            np.zeros((self.rows,self.cols), dtype=np.float32)

            pass

        return np_arr ^

    fn _transpose_order(self) -> DTypePointer[dtype]:
        var new_ptr = DTypePointer[dtype].alloc(self.rows * self.cols)
        
        for idx_col in range(self.cols):
            var tmp_ptr = self._mat_ptr.offset(idx_col)

            @parameter
            fn convert[nelts: Int](iv: Int) -> None:
                new_ptr.store[width=nelts](
                    iv + idx_col * self.rows,
                    tmp_ptr.simd_strided_load[width=nelts](self.cols),
                )
                tmp_ptr = tmp_ptr.offset(nelts * self.cols)

            vectorize[convert, simd_width](self.rows)
        return new_ptr

    