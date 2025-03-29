class Dependencies:
    """
    A centralized dependency injection container to break circular imports.
    
    This singleton class provides a simple service locator pattern implementation
    that allows components to register and retrieve dependencies without creating
    circular import references between modules.
    """
    _instance = None
    _dependencies = {}

    def __new__(cls):
        if not cls._instance:
            cls._instance = super().__new__(cls)
        return cls._instance

    @classmethod
    def register(cls, key, dependency):
        """
        Register a dependency.
        
        Args:
            key: Unique identifier for the dependency
            dependency: The object to store
        """
        cls._dependencies[key] = dependency

    @classmethod
    def get(cls, key):
        """
        Retrieve a dependency.
        
        Args:
            key: Unique identifier for the dependency
            
        Returns:
            The requested dependency object, or None if not found
        """
        return cls._dependencies.get(key)

dependencies = Dependencies()
