from algorithm import vectorize
from memory.memory import memcpy,memset_zero
from python import Python,PythonObject
from random import rand
from sys.info import simdwidthof


from .mo_num import add, sub, mul, div

struct MoVector[dtype: DType, simd_width: Int](
    Stringable, CollectionElement, Sized
):
    var _vec_ptr: UnsafePointer[Scalar[dtype]]
    var size: Int

    fn __init__(inout self, size: Int):
        self.size = size
        self._vec_ptr = UnsafePointer[Scalar[dtype]].alloc(size)
        memset_zero(self._vec_ptr, self.size)

    fn __init__(inout self, size: Int, val: Scalar[dtype]):
        self.size = size
        self._vec_ptr = UnsafePointer[Scalar[dtype]].alloc(size)

        @parameter
        fn _set_val[width: Int](iv: Int) -> None:
            self._vec_ptr.store[width=width](iv, val)

        vectorize[_set_val, simd_width](self.size)

    fn __init__(inout self, size: Int, *data: Scalar[dtype]):
        self.size = size
        self._vec_ptr = UnsafePointer[Scalar[dtype]].alloc(self.size)
        for i in range(self.size):
            self._vec_ptr[i] = data[i]

    fn __init__(inout self, size: Int, ptr: UnsafePointer[Scalar[dtype]]):
        self.size = size
        self._vec_ptr = ptr

    fn __init__(inout self, list: List[Scalar[dtype]]):
        self.size = len(list)
        self._vec_ptr = UnsafePointer[Scalar[dtype]].alloc(self.size)
        for i in range(self.size):
            self._vec_ptr[i] = list[i]
        
    fn __copyinit__(inout self, other: Self):
        self.size = other.size
        self._vec_ptr = UnsafePointer[Scalar[dtype]].alloc(self.size)
        memcpy(self._vec_ptr, other._vec_ptr, self.size)

    fn __moveinit__(inout self, owned existing: Self):
        self._vec_ptr = existing._vec_ptr
        self.size = existing.size
        existing.size = 0
        existing._vec_ptr = UnsafePointer[Scalar[dtype]]()

    fn __del__(owned self):
        self._vec_ptr.free()

    @always_inline
    fn __setitem__(inout self, idx: Int, val: Scalar[dtype]):
        self._vec_ptr.store[width=1](idx, val)

    @always_inline
    fn __getitem__(self, idx: Int) -> SIMD[dtype, 1]:
        return self._vec_ptr.load(idx)

    @always_inline
    fn __mul__(self, val: Scalar[dtype]) -> Self:
        return self._elemwise_scalar_math[mul](val)

    @always_inline
    fn __mul__(self, vec: Self) -> Self:
        return self._elemwise_vec_vec[mul](vec)

    @always_inline
    fn __add__(self, vec: Self) -> Self:
        return self._elemwise_vec_vec[add](vec)

    @always_inline
    fn __add__(self, val: Scalar[dtype]) -> Self:
        return self._elemwise_scalar_math[add](val)

    @always_inline
    fn __sub__(self, vec: Self) -> Self:
        return self._elemwise_vec_vec[sub](vec)

    @always_inline
    fn __sub__(self, val: Scalar[dtype]) -> Self:
        return self._elemwise_scalar_math[sub](val)

    @always_inline
    fn __iadd__(inout self, other: Self):
        @parameter
        fn tensor_tensor_vectorize[simd_width: Int](idx: Int) -> None:
            self._vec_ptr.store[width=simd_width](
                idx,
                self._vec_ptr.load[width=simd_width](idx)
                + other._vec_ptr.load[width=simd_width](idx),
            )

        vectorize[tensor_tensor_vectorize, simd_width](len(self))

    @always_inline
    fn __isub__(inout self, other: Self):
        return self.__iadd__(-other)

    fn __radd__(self, val: Scalar[dtype]) -> Self:
        return self.__add__(val)

    fn __rsub__(self, val: Scalar[dtype]) -> Self:
        return (-self).__add__(val)

    fn __rmul__(self, val: Scalar[dtype]) -> Self:
        return self.__mul__(val)

    fn __matmul__(self, vec: Self) -> Scalar[dtype]:
        return MoNum[dtype, simd_width].sum(self * vec)

    @always_inline
    fn __truediv__(self, s: Scalar[dtype]) -> Self:
        return self._elemwise_scalar_math[div](s)

    @always_inline
    fn __truediv__(self, vec: Self) -> Self:
        return self._elemwise_vec_vec[div](vec)

    @always_inline
    fn __itruediv__(inout self, s: Scalar[dtype]):
        alias simd_width = self.simd_width

        @parameter
        fn elemwise_vectorize[simd_width: Int](idx: Int) -> None:
            self._vec_ptr.store[width=simd_width](
                idx,
                div(
                    self._vec_ptr.load[width=simd_width](idx),
                    SIMD[dtype, simd_width](s),
                ),
            )

        vectorize[elemwise_vectorize, simd_width](len(self))

    @always_inline
    fn __neg__(self) -> Self:
        return self._elemwise_scalar_math[mul](-1)

    @always_inline
    fn __pow__(self, p: Int) -> Self:
        return self._elemwise_pow(p)

    @always_inline
    fn __eq__(self, val: Int) -> Self:
        alias simd_width = self.simd_width
        var bool_float_array = Self(len(self))

        @parameter
        fn elemwise_vectorize[simd_width: Int](idx: Int) -> None:
            var iters = SIMD[dtype, simd_width](0)
            bool_float_array._vec_ptr.store[width=simd_width](
                idx,
                (self._vec_ptr.load[width=simd_width](idx) == val).select(
                    iters + 1, iters
                ),
            )

        vectorize[elemwise_vectorize, self.simd_width](len(self))
        return bool_float_array

    @always_inline
    fn exp(self) -> Self:
        return self._elemwise_exp()

    @always_inline
    fn __len__(self) -> Int:
        return self.size

    @always_inline
    fn __str__(self) -> String:
        var printStr: String = "["
        for i in range(self.size):
            if i > 0:
                printStr += ", "
            printStr += str(self._vec_ptr[i])
        printStr += "]"
        return printStr

    @always_inline
    fn formated_str(self, digits: Int = 2) -> String:
        var printStr: String = "["
        for i in range(self.size):
            if i > 0:
                printStr += ", "
            printStr += format_float(self._vec_ptr[i], digits)
        printStr += "]"
        return printStr

    @staticmethod
    fn rand(size: Int) -> MoVector[dtype, simd_width]:
        var res = MoVector[dtype, simd_width](size)
        rand(res._vec_ptr, size)
        return res

    @staticmethod
    fn from_numpy(np_array: PythonObject) raises -> Self:
        var np_vec_ptr = UnsafePointer[Scalar[dtype]](
            __mlir_op.`pop.index_to_pointer`[
                _type = __mlir_type[`!kgen.pointer<scalar<`, dtype.value, `>>`]
            ](
                SIMD[DType.index, 1](
                    np_array.__array_interface__["data"][0].__index__()
                ).value
            )
        )
        var size = int(np_array.shape[0])
        var out = MoVector[dtype, simd_width](size)

        memcpy(out._vec_ptr, np_vec_ptr, size)

        return out

    '''
    fn to_numpy(self) raises -> PythonObject:
        var np = Python.import_module("numpy")
        var np_arr = np.empty((self.height,self.width), dtype='f')
        memcpy(np_arr.__array_interface__['data'][0].unsafe_get_as_pointer[DType.float32](), self.data, self.size)
        return np_arr^
    '''
    
    fn to_numpy(self) raises -> PythonObject:
        var np = Python.import_module("numpy")
       
        var type = np.float32
        if dtype == DType.float64:
            type = np.float64
        elif dtype == DType.float16:
            type = np.float16

        var np_vec = np.zeros(self.size, dtype=type)

        '''
        var np_vec_ptr = UnsafePointer[Scalar[dtype]](
            __mlir_op.`pop.index_to_pointer`[
                _type = __mlir_type[`!kgen.pointer<scalar<`, dtype.value, `>>`]
            ](
                SIMD[DType.index, 1](
                    np_vec.__array_interface__["data"][0].__index__()
                ).value
            )
        )
        '''

        memcpy(np_vec.__array_interface__['data'][0].unsafe_get_as_pointer[dtype](), self._vec_ptr, len(self))
       
        #memcpy(np_vec_ptr, self._vec_ptr, len(self))

        return np_vec^
    

    @always_inline
    fn zero(inout self) -> None:
        memset_zero(self._vec_ptr, self.size)

    @always_inline
    fn load[width: Int](self, idx: Int) -> SIMD[dtype, width]:
        return self._vec_ptr.load[width=width](idx)

    @always_inline
    fn store[width: Int](self, idx: Int, val: SIMD[dtype, width]):
        return self._vec_ptr.store[width=width](idx, val)

    @always_inline
    fn _elemwise_scalar_math[
        func: fn[dtype: DType, width: Int] (
            SIMD[dtype, width], SIMD[dtype, width]
        ) -> SIMD[dtype, width]
    ](self, s: Scalar[dtype]) -> Self:
        alias simd_width: Int = self.simd_width
        var new_vect = Self(self.size)

        @parameter
        fn elemwise_vectorize[simd_width: Int](idx: Int) -> None:
            new_vect._vec_ptr.store[width=simd_width](
                idx,
                func[dtype, simd_width](
                    self._vec_ptr.load[width=simd_width](idx),
                    SIMD[dtype, simd_width](s),
                ),
            )
        vectorize[elemwise_vectorize, simd_width](self.size)
        return new_vect

    @always_inline
    fn _elemwise_vec_vec[
        func: fn[dtype: DType, width: Int] (
            SIMD[dtype, width], SIMD[dtype, width]
        ) -> SIMD[dtype, width]
    ](self, vec: Self) -> Self:
        alias simd_width: Int = self.simd_width
        var new_vect = Self(self.size)

        @parameter
        fn tensor_tensor_vectorize[simd_width: Int](idx: Int) -> None:
            new_vect._vec_ptr.store[width=simd_width](
                idx,
                func[dtype, simd_width](
                    self._vec_ptr.load[width=simd_width](idx),
                    vec._vec_ptr.load[width=simd_width](idx),
                ),
            )

        vectorize[tensor_tensor_vectorize, simd_width](self.size)
        return new_vect

    @always_inline
    fn _elemwise_pow(self, p: Int) -> Self:
        alias simd_width = self.simd_width
        var new_vect = Self(self.size)

        @parameter
        fn pow_vectorize[simd_width: Int](idx: Int) -> None:
            new_vect._vec_ptr.store[width=simd_width](
                idx, pow(self._vec_ptr.load[width=simd_width](idx), p)
            )

        vectorize[pow_vectorize, simd_width](self.size)
        return new_vect

    @always_inline
    fn _elemwise_exp(self) -> Self:
        alias simd_width = self.simd_width
        var new_vect = Self(self.size)

        @parameter
        fn exp_vectorize[simd_width: Int](idx: Int) -> None:
            new_vect._vec_ptr.store[width=simd_width](
                idx, math.exp(self._vec_ptr.load[width=simd_width](idx))
            )

        vectorize[exp_vectorize, simd_width](self.size)
        return new_vect
