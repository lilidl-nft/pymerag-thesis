import inspect

import fastmcp
from fastmcp import FastMCP

print("fastmcp version:", getattr(fastmcp, "__version__", "unknown"))

# Check FastMCP methods
for name in dir(FastMCP):
    if not name.startswith('_'):
        attr = getattr(FastMCP, name)
        if callable(attr):
            try:
                sig = inspect.signature(attr)
                print(f"  {name}{sig}")
            except Exception:
                print(f"  {name}(...)")
        else:
            print(f"  {name} = {type(attr).__name__}")

# Check tool decorator
mcp = FastMCP('test')
print("\ntool type:", type(mcp.tool))
print("tool callable:", callable(mcp.tool))

# Check server methods
print("\nServer methods:")
for name in dir(mcp):
    if not name.startswith('_'):
        attr = getattr(mcp, name)
        if callable(attr) and 'run' in name or 'serve' in name or 'start' in name:
            print(f"  {name}")
