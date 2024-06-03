from time.time import sleep, now
from mopro.bar_printer import BarPrinter, BarSettings
from mopro.utils import (
    format_float,
    format_seconds,
    int_to_padded_string,
    mult_string,
)


fn vectorize_bar[
    callback: fn[Int] (Int, /) capturing -> None, nelts: Int
](
    total: Int,
    prefix: String = "",
    postfix: String = "",
    bar_size: Int = 50,
    bar_fill: String = "█",
    bar_empty: String = "░",
):
    fn _f[nelts: Int](iv: Int, inout bs: BarSettings, /) capturing -> Bool:
        callback[nelts](iv)
        return True

    vectorize_bar[_f, nelts](
        total, prefix, postfix, bar_size, bar_fill, bar_empty
    )


fn vectorize_bar[
    callback: fn[Int] (Int, /) capturing -> Bool, nelts: Int
](
    total: Int,
    prefix: String = "",
    postfix: String = "",
    bar_size: Int = 50,
    bar_fill: String = "█",
    bar_empty: String = "░",
):
    fn _f[nelts: Int](iv: Int, inout bs: BarSettings, /) capturing -> Bool:
        return callback[nelts](iv)

    vectorize_bar[_f, nelts](
        total, prefix, postfix, bar_size, bar_fill, bar_empty
    )


fn vectorize_bar[
    callback: fn[Int] (Int, inout BarSettings, /) capturing -> None, nelts: Int
](
    total: Int,
    prefix: String = "",
    postfix: String = "",
    bar_size: Int = 50,
    bar_fill: String = "█",
    bar_empty: String = "░",
):
    fn _f[nelts: Int](iv: Int, inout bs: BarSettings, /) capturing -> Bool:
        callback[nelts](iv, bs)
        return True

    vectorize_bar[_f, nelts](
        total, prefix, postfix, bar_size, bar_fill, bar_empty
    )


fn vectorize_bar[
    callback: fn[Int] (Int, inout BarSettings, /) capturing -> Bool, nelts: Int
](
    total: Int,
    prefix: String = "",
    postfix: String = "",
    bar_size: Int = 50,
    bar_fill: String = "█",
    bar_empty: String = "░",
):
    """
    A simple progress bar.

    Parameters:
        callback: Function to call in each iteration.
        nelts: Numder of elements for the vectorized operations.

    Args:
        total: The number of iterations.
        prefix: Prefix string to display before the progress bar. (default: '')
        bar_size: The size of the progress bar. (default: 50)
        bar_fill: Bar fill character.  (default: "█")
        bar_empty: Bar empty character. (default: "░")
    """

    var bar_printer = BarPrinter(
        total, BarSettings(prefix, postfix, bar_size, bar_fill, bar_empty)
    )

    bar_printer.print(0)

    for step in range(total // nelts):
        if not callback[nelts](step * nelts, bar_printer.bar_settings):
            break
        bar_printer.print(step * nelts + 1)

    for i in range(total % nelts):
        var step = (total // nelts) * nelts + i
        if not callback[1](step, bar_printer.bar_settings):
            break
        bar_printer.print(step + 1)
