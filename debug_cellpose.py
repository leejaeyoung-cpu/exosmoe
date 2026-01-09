import cellpose
print(f"Cellpose file: {cellpose.__file__}")
print(f"Dir cellpose: {dir(cellpose)}")
try:
    from cellpose import models
    print(f"Dir models: {dir(models)}")
except Exception as e:
    print(f"Error importing models: {e}")
