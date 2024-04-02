from colorama import Fore, Style

class ColorPrinter:
    """Object that prints messages with alternating colors."""

    _COLORS = [
        Fore.LIGHTBLACK_EX,
        Fore.BLUE,
        Fore.CYAN,
        Fore.GREEN,
        Fore.MAGENTA,
        Fore.RED,
        Fore.YELLOW,
    ]

    def __init__(self) -> None:
        self.reset()

    def __call__(self, msg: str, end: str = '\n') -> None:
        """Print message with alternating colors."""
        print(
            self._COLORS[self._color_index] + str(msg) + Style.RESET_ALL,
            end=end,
        )
        self._color_index = (self._color_index + 1) % len(self._COLORS)

    def reset(self) -> None:
        """Reset color index to 0."""
        self._color_index = 0
