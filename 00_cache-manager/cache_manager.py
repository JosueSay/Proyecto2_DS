import os
import yaml

class CacheManager:
    def __init__(self, configPath: str, cacheDir: str):
        # crea el directorio de cache si no existe
        self.cacheDir = cacheDir
        os.makedirs(self.cacheDir, exist_ok=True)

        # carga la config yaml con la lista de archivos de cache
        with open(configPath, "r") as f:
            self.config = yaml.safe_load(f)
        
        # obtiene el mapeo de claves -> nombres de archivo
        self.cacheFiles = self.config.get("cache_files", {})

    def getCachePath(self, key: str) -> str:
        # busca el archivo asociado a la clave en la config
        filename = self.cacheFiles.get(key)
        if not filename:
            raise KeyError(f"No existe la clave de cache '{key}' en config")
        return os.path.join(self.cacheDir, filename)

    def exists(self, key: str) -> bool:
        # verifica si el archivo de cache ya fue creado
        return os.path.exists(self.getCachePath(key))

    def create(self, key: str, content: str = ""):
        # crea el archivo de cache y guarda el contenido
        path = self.getCachePath(key)
        with open(path, "w") as f:
            f.write(content)

    def remove(self, key: str):
        # elimina el archivo de cache si existe
        path = self.getC
