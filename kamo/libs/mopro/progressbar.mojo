from time.time import sleep, now
from .bar_printer import BarPrinter, BarSettings
from .utils import (
    format_float,
    format_seconds,
    int_to_padded_string,
    mult_string,
)


fn progress_bar[
    callback: fn (Int, /) capturing -> None
](
    total: Int,
    prefix: String = "",
    postfix: String = "",
    bar_size: Int = 50,
    bar_fill: String = "█",
    bar_empty: String = "░",
):
    fn _f(i: Int, inout bs: BarSettings, /) capturing -> Bool:
        callback(i)
        return True

    progress_bar[_f](total, prefix, postfix, bar_size, bar_fill, bar_empty)


fn progress_bar[
    callback: fn (Int, /) capturing -> Bool
](
    total: Int,
    prefix: String = "",
    postfix: String = "",
    bar_size: Int = 50,
    bar_fill: String = "█",
    bar_empty: String = "░",
):
    fn _f(i: Int, inout bs: BarSettings, /) capturing -> Bool:
        return callback(i)

    progress_bar[_f](total, prefix, postfix, bar_size, bar_fill, bar_empty)


fn progress_bar[
    callback: fn (Int, inout BarSettings, /) capturing -> None
](
    total: Int,
    prefix: String = "",
    postfix: String = "",
    bar_size: Int = 50,
    bar_fill: String = "█",
    bar_empty: String = "░",
):
    fn _f(i: Int, inout bs: BarSettings, /) capturing -> Bool:
        callback(i, bs)
        return True

    progress_bar[_f](total, prefix, postfix, bar_size, bar_fill, bar_empty)


fn progress_bar[
    callback: fn (Int, inout BarSettings, /) capturing -> Bool
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

    Args:
        total: The number of iterations.
        prefix: Prefix string to display before the progress bar. (default: '')
        postfix: Postfix string to display after the progress bar. (default: '')
        bar_size: The size of the progress bar. (default: 50)
        bar_fill: Bar fill character.  (default: "█")
        bar_empty: Bar empty character. (default: "░")
    """

    var bar_printer = BarPrinter(
        total, BarSettings(prefix, postfix, bar_size, bar_fill, bar_empty)
    )

    bar_printer.print(0)
    for step in range(total):
        if not callback(step, bar_printer.bar_settings):
            break
        bar_printer.print(step + 1)
