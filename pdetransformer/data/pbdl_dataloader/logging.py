HEADER = "\033[95m"
DARKGREY = "\033[38;5;240m"
OKBLUE = "\033[94m"
OKGREEN = "\033[92m"
SUCCESS_CYAN = "\033[96m"
WARN_YELLOW = "\033[93m"
FAIL_RED = "\033[91m"
ENDC = "\033[0m"
BOLD = "\033[1m"
R_BOLD = "\033[22m"
UNDERLINE = "\033[4m"


def info(msg: str):
    print(BOLD + "Info:" + R_BOLD + " " + msg + ENDC)

def success(msg: str):
    print(SUCCESS_CYAN + BOLD + "Success:" + R_BOLD + " " + msg + ENDC)

def warn(msg: str):
    print(WARN_YELLOW + BOLD + "Warning:" + R_BOLD + " " + msg + ENDC)

def fail(msg: str):
    print(FAIL_RED + BOLD + "Fail:" + R_BOLD + " " + msg + ENDC)

def corrupt(msg: str):
    print(FAIL_RED + BOLD + "Corrupt:" + R_BOLD + " " + msg + ENDC)