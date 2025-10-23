import os
import yaml

class CacheManager:
    def __init__(self, configPath: str, cacheDir: str):
        self.cacheDir = cacheDir
        os.makedirs(self.cacheDir, exist_ok=True)

        # Cargar YAML
        with open(configPath, "r") as f:
            self.config = yaml.safe_load(f)
        
        self.cacheFiles = self.config.get("cache_files", {})

    def getCachePath(self, key: str) -> str:
        """Devuelve la ruta completa del archivo de cache por clave."""
        filename = self.cacheFiles.get(key)
        if not filename:
            raise KeyError(f"No existe la clave de cache '{key}' en config")
        return os.path.join(self.cacheDir, filename)

    def exists(self, key: str) -> bool:
        """Revisa si el cache ya existe"""
        return os.path.exists(self.getCachePath(key))

    def create(self, key: str, content: str = ""):
        """Crea un archivo de cache con contenido opcional"""
        path = self.getCachePath(key)
        with open(path, "w") as f:
            f.write(content)

    def remove(self, key: str):
        """Elimina el cache si existe"""
        path = self.getCachePath(key)
        if os.path.exists(path):
            os.remove(path)
