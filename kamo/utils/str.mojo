from kamo import dtype,eps

alias SD = Scalar[dtype]

fn int_to_padded_string(n: Int, width: Int, pad_str: String = " ") -> String:
    var res: String = ""
    for _ in range(width - len(str(n))):
        res += pad_str
    return res + str(n)


fn str_maxlen(s:String,mlen:Int)->String:
    return s[:min(mlen,len(s))]