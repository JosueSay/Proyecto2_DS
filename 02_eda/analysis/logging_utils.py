def writeLine(f, msg=""):
    f.write(msg + "\n")

def pct(x):
    return f"{float(x):.2f}%" if x is not None else "0.00%"
