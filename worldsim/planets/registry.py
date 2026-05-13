from weakref import WeakValueDictionary

# Global mapping of planet names to planet instances
# Use weak references so unreferenced planets can be garbage collected.
PLANETS: 'WeakValueDictionary[str, "Planet"]' = WeakValueDictionary()
