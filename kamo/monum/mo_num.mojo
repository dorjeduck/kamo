from algorithm import vectorize
from algorithm.functional import elementwise
from math import mul, div, sin,exp,mod,pow, add, trunc, align_down, align_down_residual
from random import rand


struct MoNum[dtype:DType,simd_width:Int]:


    @staticmethod
    fn sin(mv:MoVector[dtype,simd_width]) -> MoVector[dtype,simd_width]:
        return MoNum[dtype,simd_width]._elemwise_func_1[sin](mv)
    
    @staticmethod
    fn cos(mv:MoVector[dtype,simd_width]) -> MoVector[dtype,simd_width]:
        return MoNum[dtype,simd_width]._elemwise_func_1[cos](mv)

    @staticmethod
    fn sin(mm:MoMatrix[dtype,simd_width]) -> MoMatrix[dtype,simd_width]:
        return MoNum[dtype,simd_width]._elemwise_func_1[sin](mm)
    
    @staticmethod
    fn cos(mm:MoMatrix[dtype,simd_width]) -> MoMatrix[dtype,simd_width]:
        return MoNum[dtype,simd_width]._elemwise_func_1[cos](mm)

    @staticmethod
    fn exp(mm:MoMatrix[dtype,simd_width]) -> MoMatrix[dtype,simd_width]:
        return MoNum[dtype,simd_width]._elemwise_func_1[exp](mm)
    
    @staticmethod
    fn pow(mm:MoMatrix[dtype,simd_width],p:Scalar[dtype]) -> MoMatrix[dtype,simd_width]:
        return MoNum[dtype,simd_width]._elemwise_func_2[pow](mm,p)

    @staticmethod
    fn linspace(start:Scalar[dtype], stop:Scalar[dtype], size:Int) -> MoVector[dtype,simd_width]:
        
        var res = MoVector[dtype,simd_width](size)
        var step = (stop - start) / (size - 1)

        for i in range(size-1):
            res[i] = start+i*step
        res[size-1] = stop

        return res

    @staticmethod
    fn linspace(
        start: Scalar[dtype], stop: Scalar[dtype], rows: Int, cols: Int
    ) -> MoMatrix[dtype,simd_width]:
       
        var res = MoMatrix[dtype,simd_width](rows, cols)

        var size = rows * cols
        var step:Scalar[dtype] = (stop - start) / (size - 1)
        
        for i in range(size - 1):
            res[i] = start + i * step
        res[size - 1] = stop

        return res

    @staticmethod
    fn _elemwise_scalar_math[
        func: fn[dtype: DType, width: Int] (
            SIMD[dtype, width], SIMD[dtype, width]
        ) -> SIMD[dtype, width]
    ](mm:MoMatrix[dtype,simd_width], s: Scalar[dtype]) -> MoMatrix[dtype,simd_width]:
       
        var new_mat = MoMatrix[dtype,simd_width](mm.rows, mm.cols)
        @parameter
        fn elemwise_vectorize[nelts: Int](iv: Int) -> None:
            new_mat._mat_ptr.store[width=nelts](
                iv,
                func[dtype, nelts](
                    mm._mat_ptr.load[width=nelts](iv),
                    SIMD[dtype, nelts](s),
                ),
            )
        vectorize[elemwise_vectorize, simd_width](mm.rows * mm.cols)
        
        return new_mat

    @staticmethod
    fn _elemwise_matrix_matrix[
        func: fn[dtype: DType, width: Int] (
            SIMD[dtype, width], SIMD[dtype, width]
        ) -> SIMD[dtype, width]
    ](mm:MoMatrix[dtype,simd_width], mat: MoMatrix[dtype,simd_width]) -> MoMatrix[dtype,simd_width]:
       
        var new_mat = MoMatrix[dtype,simd_width](mm.rows, mm.cols)

        @parameter
        fn matrix_matrix_vectorize[nelts: Int](iv: Int) -> None:
            new_mat._mat_ptr.store[width=nelts](
                iv,
                func[dtype, nelts](
                    mm._mat_ptr.load[width=nelts](iv),
                    mat._mat_ptr.load[width=nelts](iv),
                ),
            )

        vectorize[matrix_matrix_vectorize, simd_width](len(mm))
        return new_mat

    @staticmethod
    fn _elemwise_func_1[
        func: fn[dtype: DType, width: Int] (
            SIMD[dtype, width]
        ) -> SIMD[dtype, width] 
        ](mv:MoVector[dtype,simd_width]) -> MoVector[dtype,simd_width]:
       

        var new_vec = MoVector[dtype,simd_width](mv.size)
        @parameter
        fn _fun[nelts: Int](iv: Int) -> None:
            new_vec._vec_ptr.store[width=nelts](iv, func(mv._vec_ptr.load[width=nelts](iv)))
        vectorize[_fun, simd_width](mv.size)
        return new_vec
    
    @staticmethod
    fn _elemwise_func_1[
        func: fn[dtype: DType, width: Int] (
            SIMD[dtype, width]
        ) -> SIMD[dtype, width] 
        ](mm:MoMatrix[dtype,simd_width]) -> MoMatrix[dtype,simd_width]:
       

        var new_mat = MoMatrix[dtype,simd_width](mm.rows,mm.cols)
        @parameter
        fn _fun[nelts: Int](iv: Int) -> None:
            new_mat._mat_ptr.store[width=nelts](iv, func(mm._mat_ptr.load[width=nelts](iv)))
        vectorize[_fun, simd_width](mm.rows*mm.cols)
        return new_mat

    @staticmethod
    fn _elemwise_func_2[
        func: fn[dtype: DType, width: Int] (
            SIMD[dtype, width], SIMD[dtype, width]
        ) -> SIMD[dtype, width] 
        ](mm:MoMatrix[dtype,simd_width],p:Scalar[dtype]) -> MoMatrix[dtype,simd_width]:
       
        var new_mat = MoMatrix[dtype,simd_width](mm.rows,mm.cols)
        @parameter
        fn _fun[nelts: Int](iv: Int) -> None:
            new_mat._mat_ptr.store[width=nelts](iv, func(mm._mat_ptr.load[width=nelts](iv), SIMD[dtype, nelts](p)))
        vectorize[_fun, simd_width](len(mm))
        return new_mat

     
    