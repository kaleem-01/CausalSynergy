import os

def clear_console_soft():
    print("\033[2J\033[H", end="")  # Clears screen and moves cursor to top


def print_banner(msg: str, color: str = "cyan"):
    colors = {
        "red": "\033[91m", "green": "\033[92m", "yellow": "\033[93m",
        "blue": "\033[94m", "magenta": "\033[95m", "cyan": "\033[96m", "white": "\033[97m",
    }
    reset = "\033[0m"
    width = os.get_terminal_size().columns
    print(f"\n{colors.get(color, '')}{'═' * width}")
    print(f"{msg.center(width)}")
    print(f"{'═' * width}{reset}\n")