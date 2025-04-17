import sys 
deps = ["PyQt5", "PySide6", "numpy", "matplotlib"] 
missing = [] 
for dep in deps: 
    try: 
        __import__(dep) 
        print(f"{dep}: OK") 
    except ImportError: 
        missing.append(dep) 
        print(f"{dep}: MISSING") 
if missing: 
    print(f"Missing dependencies: {', '.join(missing)}") 
    sys.exit(1) 
else: 
    print("All dependencies satisfied") 
    sys.exit(0) 
