from algorithm import vectorize
from math import (
    mul,
    pow,
    div,
    exp,
    mod,
    add,
    sub,
    trunc,
    align_down,
    align_down_residual,
)
from random import rand
from python import Python


struct MoVector[
    dtype: DType = DType.float32, simd_width: Int = 2 * simdwidthof[dtype]()
](Stringable, CollectionElement, Sized):
    var _vec_ptr: DTypePointer[dtype]
    var size: Int

    @always_inline
    fn __init__(inout self, size: Int):
        self.size = size
        self._vec_ptr = DTypePointer[dtype].alloc(size)

        memset_zero[dtype](self._vec_ptr, self.size)

    fn __init__(inout self, size: Int, val: Scalar[dtype]):
        self.size = size
        self._vec_ptr = DTypePointer[dtype].alloc(size)

        @parameter
        fn splat_val[width: Int](iv: Int) -> None:
            self._vec_ptr.store[width=width](
                iv, self._vec_ptr.load[width=width](iv).splat(val)
            )

        vectorize[splat_val, simd_width](self.size)
    
    @always_inline
    fn __init__(inout self, size: Int, *data: Scalar[dtype]):
        
        self.size = size
        self._vec_ptr = DTypePointer[dtype].alloc(self.size)
        for i in range(self.size):
            self._vec_ptr[i] = data[i]

    @always_inline
    fn __init__(inout self, size: Int, ptr: DTypePointer[dtype]):
        
        self.size = size
        self._vec_ptr = DTypePointer[dtype].alloc(self.size)
        for i in range(self.size):
            self._vec_ptr[i] = ptr[i]

    fn __init__(inout self, owned list: List[Scalar[dtype]]):
        self.size = len(list)
        self._vec_ptr = DTypePointer[dtype].alloc(self.size)
        for i in range(self.size):
            self._vec_ptr[i] = list[i]

    

    @always_inline
    fn __copyinit__(inout self, other: Self):
        self.size = other.size
        self._vec_ptr = DTypePointer[dtype].alloc(self.size)
        memcpy(self._vec_ptr, other._vec_ptr, self.size)

    fn __moveinit__(inout self, owned existing: Self):
        self._vec_ptr = existing._vec_ptr
        self.size = existing.size
        existing.size = 0
        existing._vec_ptr = DTypePointer[dtype]()

    fn __del__(owned self):
        if self.size > 0:
            self._vec_ptr.free()

    @always_inline
    fn __setitem__(inout self, idx:Int, val: Scalar[dtype]):
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
        return (self*vec).sum()

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
    fn sum(self) -> Scalar[dtype]:

        var res:Scalar[dtype] = 0
        @parameter
        fn _sum[nelts:Int](iv:Int):
            res += self._vec_ptr.load[width=nelts](iv).reduce_add()
        vectorize[_sum,simd_width](self.size)
        return res

    @always_inline
    fn __len__(self) -> Int:
        return self.size

    fn __str__(self) -> String:
        var printStr: String = "["
        for i in range(self.size):
            if i > 0:
                printStr += ", "
            printStr += self._vec_ptr[i]
        printStr += "]"
        return printStr

    #fn diag(self):
    #    var result = MoMatrix[dtype](self.size, self.size, 0)
    
    fn to_momatrix(inout self,num_rows:Int) -> MoMatrix[dtype]:
        if self.size/num_rows != self.size//num_rows:
            print("problem with transform to matrix")
        var res = MoMatrix[dtype](num_rows,self.size//num_rows,self._vec_ptr)
        return res
    
    @staticmethod
    fn rand(size: Int) -> Self:
        var _vec_ptr = DTypePointer[dtype].alloc(size)
        rand(_vec_ptr, size)
        return Self(size, _vec_ptr)

    @staticmethod
    fn from_numpy(np_array: PythonObject) raises -> Self:
        var np_vec_ptr = DTypePointer[dtype](
        __mlir_op.`pop.index_to_pointer`[
            _type = __mlir_type[`!kgen.pointer<scalar<`, dtype.value, `>>`]
        ](
            SIMD[DType.index,1](np_array.__array_interface__['data'][0].__index__()).value
        )
    )
        var size = int(np_array.shape[0])
       
        var _vec_ptr = DTypePointer[dtype].alloc(size)
        memcpy(_vec_ptr, np_vec_ptr, size)
        var out = Self(size,_vec_ptr)
        return out ^

    fn to_numpy(self) raises -> PythonObject:
        var np = Python.import_module("numpy")
        var np_vec = np.zeros(self.size)
        var np_vec_ptr = DTypePointer[dtype](
        __mlir_op.`pop.index_to_pointer`[
            _type = __mlir_type[`!kgen.pointer<scalar<`, dtype.value, `>>`]
        ](
            SIMD[DType.index,1](np_vec.__array_interface__['data'][0].__index__()).value
        )
    )
        memcpy(np_vec_ptr, self._vec_ptr, len(self))
        return np_vec ^

    
        
    @always_inline
    fn zero(inout self) -> None:
        memset_zero[dtype](self._vec_ptr, self.size)


    @always_inline
    fn load[width: Int](self, idx: Int) -> SIMD[dtype, width]:
        return self._vec_ptr.load[width=width](idx)

    @always_inline
    fn store[width: Int](self, idx:Int, val: SIMD[dtype, width]):
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
                idx, exp(self._vec_ptr.load[width=simd_width](idx))
            )

        vectorize[exp_vectorize, simd_width](self.size)
        return new_vect
