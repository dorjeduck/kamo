from algorithm import vectorize
from algorithm.functional import elementwise
from math import  log,tanh, sin,cos,exp, trunc, align_down
from memory.memory import memcpy
from random import rand
from utils.numerics import isnan

alias PI = 3.141592653589793

fn format_float[dtype:DType](f: Scalar[dtype], digits: Int = 2) -> String:
    if abs(f - 0) < eps:
        return "0"
    var sign = "-" if f<0 else ""
    var ff = str(int(round(abs(f * 10**digits))))
    var l = len(ff)
    var pd = ff[: l - digits] if abs(l-digits) > 0 else "0"  
    return sign + pd + "." + ff[l - digits :]

fn add[dtype: DType, width: Int](x1:SIMD[dtype, width],x2:SIMD[dtype, width])->SIMD[dtype, width]:
    return x1+x2

fn sub[dtype: DType, width: Int](x1:SIMD[dtype, width],x2:SIMD[dtype, width])->SIMD[dtype, width]:
    return x1-x2

fn mul[dtype: DType, width: Int](x1:SIMD[dtype, width],x2:SIMD[dtype, width])->SIMD[dtype, width]:
    return x1*x2

fn div[dtype: DType, width: Int](x1:SIMD[dtype, width],x2:SIMD[dtype, width])->SIMD[dtype, width]:
    return x1/x2


struct MoNum[dtype:DType,simd_width:Int]:

    @staticmethod
    fn split(s:String,sep:String,maxsplit: Int = -1) raises -> MoVector[dtype,simd_width]:
        var str_list = s.split(sep,maxsplit)

        var res = MoVector[dtype,simd_width](str_list.size)

        for i in range(str_list.size):
            res[i] = atof(str_list[i]).cast[dtype]()
        
        return res

    @staticmethod
    fn nantozero(mv:MoVector[dtype,simd_width]):
        for i in range(mv.size):
            if isnan(mv._vec_ptr[i]):
                mv._vec_ptr[i] = 0

    @staticmethod
    fn nantozero(mm:MoMatrix[dtype,simd_width]):
        for i in range(mm.rows*mm.cols):
            if isnan(mm._mat_ptr[i]):
                mm._mat_ptr[i] = 0

    @staticmethod
    fn inplace_copy(target:MoVector[dtype,simd_width],source:MoVector[dtype,simd_width]):
        memcpy(target._vec_ptr, source._vec_ptr, source.size)
    
    @staticmethod
    fn inplace_copy(target:MoMatrix[dtype,simd_width],source:MoMatrix[dtype,simd_width]):
        memcpy(target._mat_ptr, source._mat_ptr, source.cols*source.rows)

    @staticmethod
    fn minmax(mv:MoVector[dtype,simd_width]) -> (Scalar[dtype],Scalar[dtype]):
        return Self._minmax(mv._vec_ptr,mv.size)
       
    @staticmethod
    fn minmax(mm:MoMatrix[dtype,simd_width]) -> (Scalar[dtype],Scalar[dtype]):
        return Self._minmax(mm._mat_ptr,mm.rows*mm.cols)
   
    @staticmethod
    fn sum(mv:MoVector[dtype,simd_width]) -> Scalar[dtype]:
        var res:Scalar[dtype] = 0
        @parameter
        fn _sum[width:Int](iv:Int):
            res += mv._vec_ptr.load[width=width](iv).reduce_add()
        vectorize[_sum,simd_width](mv.size)
        return res

    @staticmethod
    fn sum( mm:MoMatrix[dtype,simd_width], axis: Int) -> MoVector[dtype,simd_width]:
        if axis == 0:
            ##todo optimize
            var res = MoVector[dtype,simd_width](mm.cols)
            for j in range(mm.cols):
                for i in range(mm.rows):
                    res[j] += mm[i, j]
            return res
        else:
            var res = MoVector[dtype,simd_width](mm.rows)
            for i in range(mm.rows):

                @parameter
                fn _sum_row[width: Int](iv: Int):
                    res[i] += mm.load[width](i, iv).reduce_add()

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
    fn log(mv:MoVector[dtype,simd_width]) -> MoVector[dtype,simd_width]:
        return MoNum[dtype,simd_width]._elemwise_func_1[log](mv)

    @staticmethod
    fn sin(mm:MoMatrix[dtype,simd_width]) -> MoMatrix[dtype,simd_width]:
        return MoNum[dtype,simd_width]._elemwise_func_1[sin](mm)
    
    @staticmethod
    fn cos(mm:MoMatrix[dtype,simd_width]) -> MoMatrix[dtype,simd_width]:
        return MoNum[dtype,simd_width]._elemwise_func_1[cos](mm)

    @staticmethod
    fn log(mm:MoMatrix[dtype,simd_width]) -> MoMatrix[dtype,simd_width]:
        return MoNum[dtype,simd_width]._elemwise_func_1[log](mm)

    @staticmethod
    fn exp(ms:Scalar[dtype]) -> Scalar[dtype]:
        return exp(ms)

    @staticmethod
    fn exp(mv:MoVector[dtype,simd_width]) -> MoVector[dtype,simd_width]:
        return MoNum[dtype,simd_width]._elemwise_func_1[exp](mv)

    @staticmethod
    fn exp(mm:MoMatrix[dtype,simd_width]) -> MoMatrix[dtype,simd_width]:
        return MoNum[dtype,simd_width]._elemwise_func_1[exp](mm)
    
    @staticmethod
    fn tanh(ms:Scalar[dtype]) -> Scalar[dtype]:
        return tanh(ms)
    
    @staticmethod
    fn tanh(mv:MoVector[dtype,simd_width]) -> MoVector[dtype,simd_width]:
        return MoNum[dtype,simd_width]._elemwise_func_1[tanh](mv)
    
    @staticmethod
    fn pow(ms:Scalar[dtype],p:Scalar[dtype]=2) -> Scalar[dtype]:
        return pow(ms,p)
   
    @staticmethod
    fn pow(mv:MoVector[dtype,simd_width],p:Scalar[dtype]=2) -> MoVector[dtype,simd_width]:
        return MoNum[dtype,simd_width]._elemwise_func_pow(mv,p)
    
    @staticmethod
    fn pow(mm:MoMatrix[dtype,simd_width],p:Scalar[dtype]=2) -> MoMatrix[dtype,simd_width]:
        return MoNum[dtype,simd_width]._elemwise_func_pow(mm,p)
    
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
        res[size-1] = stop
        
        return res
    

    @staticmethod
    fn linspace2D(start:Scalar[dtype], stop:Scalar[dtype], rows:Int,cols:Int,axis:Int=0) -> MoMatrix[dtype,simd_width]:
       
        var res = MoMatrix[dtype,simd_width](rows,cols)

        if axis == 0:

            var step = (stop - start) / (cols - 1)

            for i in range(rows):
                for j in range(cols-1):
                    res[i,j] = start+j*step
                res[i,cols-1] = stop
                
        else:
            var step = (stop - start) / (rows - 1)

            for j in range(cols):
                for i in range(rows-1): 
                    res[i,j] = start+i*step
                res[rows-1,j] = stop

        return res

    @staticmethod
    fn meshgrid(mv1:MoVector[dtype,simd_width],mv2:MoVector[dtype,simd_width])-> (MoMatrix[dtype,simd_width],MoMatrix[dtype,simd_width]):
        var res1 = MoMatrix[dtype,simd_width](mv2.size,mv1.size)
        var res2 = MoMatrix[dtype,simd_width](mv2.size,mv1.size)

        for i in range(mv2.size):
            res1.insert_row(i,mv1)
            res2.insert_row(i,(MoVector[dtype,simd_width](mv1.size,mv2[i])))

        return res1,res2
  
              
    @staticmethod
    fn arange(start: Int,stop:Int,step:Int=1) -> MoVector[dtype,simd_width]:
        var n = (stop-start-1)//step
        var res = MoVector[dtype,simd_width](n)
        for i in range(n):
            res[i] = start+i
        return res

    @staticmethod
    fn diag(mv:MoVector[dtype,simd_width])->MoMatrix[dtype,simd_width]:
        var res = MoMatrix[dtype,simd_width](mv.size,mv.size)
        for i in range(mv.size):
            res[i,i] = mv[i]
        return res
            
    @staticmethod
    fn _elemwise_scalar_math[
        func: fn[dtype: DType, width: Int] (
            SIMD[dtype, width], SIMD[dtype, width]
        ) -> SIMD[dtype, width]
    ](mm:MoMatrix[dtype,simd_width], s: Scalar[dtype]) -> MoMatrix[dtype,simd_width]:
       
        var new_mat = MoMatrix[dtype,simd_width](mm.rows, mm.cols)
        @parameter
        fn elemwise_vectorize[width: Int](iv: Int) -> None:
            new_mat._mat_ptr.store[width=width](
                iv,
                func[dtype, width](
                    mm._mat_ptr.load[width=width](iv),
                    SIMD[dtype, width](s),
                ),
            )
        vectorize[elemwise_vectorize, simd_width](mm.rows * mm.cols)
        
        return new_mat

    @staticmethod
    @always_inline
    fn _elemwise_matrix_matrix[
        func: fn[dtype: DType, width: Int] (
            SIMD[dtype, width], SIMD[dtype, width]
        ) -> SIMD[dtype, width]
    ](mm:MoMatrix[dtype,simd_width], mat: MoMatrix[dtype,simd_width]) -> MoMatrix[dtype,simd_width]:
       
        var new_mat = MoMatrix[dtype,simd_width](mm.rows, mm.cols)

        @parameter
        fn matrix_matrix_vectorize[width: Int](iv: Int) -> None:
            new_mat._mat_ptr.store[width=width](
                iv,
                func[dtype, width](
                    mm._mat_ptr.load[width=width](iv),
                    mat._mat_ptr.load[width=width](iv),
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
        fn _fun[width: Int](iv: Int) -> None:
            new_vec._vec_ptr.store[width=width](iv, func(mv._vec_ptr.load[width=width](iv)))
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
        fn _fun[width: Int](iv: Int) -> None:
            new_mat._mat_ptr.store[width=width](iv, func(mm._mat_ptr.load[width=width](iv)))
        vectorize[_fun, simd_width](mm.rows*mm.cols)
        return new_mat
    
    @staticmethod
    fn _elemwise_func_pow(mv:MoVector[dtype,simd_width],p:Scalar[dtype]) -> MoVector[dtype,simd_width]:
       
        var new_vec = MoVector[dtype,simd_width](mv.size)
        @parameter
        fn _fun[width: Int](iv: Int) -> None:
            new_vec._vec_ptr.store[width=width](iv, pow(mv._vec_ptr.load[width=width](iv), SIMD[dtype, width](p)))
        vectorize[_fun, simd_width](len(mv))
        return new_vec


    @staticmethod
    fn _elemwise_func_pow(mm:MoMatrix[dtype,simd_width],p:Scalar[dtype]) -> MoMatrix[dtype,simd_width]:
       
        var new_mat = MoMatrix[dtype,simd_width](mm.rows,mm.cols)
        @parameter
        fn _fun[width: Int](iv: Int) -> None:
            new_mat._mat_ptr.store[width=width](iv, pow(mm._mat_ptr.load[width=width](iv), SIMD[dtype, width](p)))
        
        vectorize[_fun, simd_width](len(mm))
        return new_mat

    @staticmethod
    fn _minmax(ptr:UnsafePointer[Scalar[dtype]],size:Int) -> (Scalar[dtype],Scalar[dtype]):

        var low:Scalar[dtype] = 1e1
        var high:Scalar[dtype] = -1e12

        for i in range(size):
            if ptr[i]<low:
                low=ptr[i]
            if ptr[i]>high:
                high=ptr[i]
        return low,high
