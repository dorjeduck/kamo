from algorithm import vectorize
from algorithm.functional import elementwise
from math import min,mul, div,tanh, sin,cos,exp,mod,pow, add, trunc, align_down, align_down_residual
from random import rand

alias PI = 3.141592653589793

struct MoNum[dtype:DType,simd_width:Int]:

    @staticmethod
    fn sum(mv:MoVector[dtype,simd_width]) -> Scalar[dtype]:

        var res:Scalar[dtype] = 0
        @parameter
        fn _sum[nelts:Int](iv:Int):
            res += mv._vec_ptr.load[width=nelts](iv).reduce_add()
        vectorize[_sum,simd_width](mv.size)
        return res

    @staticmethod
    fn sum( mm:MoMatrix[dtype,simd_width], axis: Int) -> MoVector[dtype,simd_width]:
        if axis == 0:
            var res = MoVector[dtype,simd_width](mm.cols)
            for j in range(mm.cols):
                for i in range(mm.rows):
                    res[j] += mm[i, j]
            return res
        else:
            var res = MoVector[dtype,simd_width](mm.rows)
            for i in range(mm.rows):

                @parameter
                fn _sum_row[nelts: Int](iv: Int):
                    res[i] += mm.load[nelts](i, iv).reduce_add()

                vectorize[_sum_row, simd_width](mm.cols)
            return res

    @staticmethod
    fn mean(mv:MoVector[dtype,simd_width]) -> Scalar[dtype]:
        return Self.sum(Self.pow(mv)/(Scalar[dtype](mv.size)))

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
    fn exp(mv:MoVector[dtype,simd_width]) -> MoVector[dtype,simd_width]:
        return MoNum[dtype,simd_width]._elemwise_func_1[exp](mv)


    @staticmethod
    fn exp(mm:MoMatrix[dtype,simd_width]) -> MoMatrix[dtype,simd_width]:
        return MoNum[dtype,simd_width]._elemwise_func_1[exp](mm)
    
    @staticmethod
    fn tanh(mv:MoVector[dtype,simd_width]) -> MoVector[dtype,simd_width]:
        return MoNum[dtype,simd_width]._elemwise_func_1[tanh](mv)
    

    @staticmethod
    fn pow(mv:MoVector[dtype,simd_width],p:Scalar[dtype]=2) -> MoVector[dtype,simd_width]:
        return MoNum[dtype,simd_width]._elemwise_func_2[pow](mv,p)
    
    @staticmethod
    fn pow(mm:MoMatrix[dtype,simd_width],p:Scalar[dtype]=2) -> MoMatrix[dtype,simd_width]:
        return MoNum[dtype,simd_width]._elemwise_func_2[pow](mm,p)
    
    @staticmethod
    fn slice(mv:MoVector[dtype,simd_width],width:Int,add_remainder:Bool=True) -> List[MoVector[dtype,simd_width]] :

        var num = mv.size//width

        var has_remainder = num*width < mv.size

        if add_remainder and has_remainder:
            num+=1

        var res = List[MoVector[dtype,simd_width]](capacity=num)

        for i in range(num):
            var w = min(width,mv.size-i*width)
            if w == width or add_remainder:
                res.append(MoVector[dtype,simd_width](w,mv._vec_ptr+(i*width)))

        return res
    
    @staticmethod
    fn linspace(start:Scalar[dtype], stop:Scalar[dtype], size:Int) -> MoVector[dtype,simd_width]:
        
        var res = MoVector[dtype,simd_width](size)
        var step = (stop - start) / (size - 1)

        for i in range(size-1):
            res[i] = start+i*step
        res[size-1] = stop

        return res

    
    @staticmethod
    fn linspace2D(start:Scalar[dtype], stop:Scalar[dtype], rows:Int,cols:Int,axis:Int=0) -> MoMatrix[dtype,simd_width]:
        
        
        if axis == 0:
            var res = MoMatrix[dtype,simd_width](rows,cols)
            var step = (stop - start) / (cols - 1)

            for i in range(rows):
                for j in range(cols-1):
                    res[i,j] = start+j*step
                res[i,cols-1] = stop

            return res
        else:
            var res = MoMatrix[dtype,simd_width](rows,cols)
            var step = (stop - start) / (rows - 1)

            for j in range(cols):
                for i in range(rows-1): 
                    res[i,j] = start+i*step
                res[rows-1,j] = stop

            return res
    


    @staticmethod
    fn rand(size: Int) -> MoVector[dtype,simd_width]:
        var res = MoVector[dtype,simd_width](size)
        rand(res._vec_ptr, size)
        return res
        
    @staticmethod
    fn arange(start: Int,stop:Int,step:Int=1) -> MoVector[dtype,simd_width]:
        var n = (stop-start-1)//step
        var res = MoVector[dtype,simd_width](n)
        for i in range(n):
            res[i] = start+i
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
        ](mv:MoVector[dtype,simd_width],p:Scalar[dtype]) -> MoVector[dtype,simd_width]:
       
        var new_vec = MoVector[dtype,simd_width](mv.size)
        @parameter
        fn _fun[nelts: Int](iv: Int) -> None:
            new_vec._vec_ptr.store[width=nelts](iv, func(mv._vec_ptr.load[width=nelts](iv), SIMD[dtype, nelts](p)))
        vectorize[_fun, simd_width](len(mv))
        return new_vec

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

     
    