import os
from core.paths import DIRS
from cache_manager import CacheManager

def getCache():
    cfg = os.path.join(DIRS["project_dir"], "00_cache-manager", "cache_config.yaml")
    return CacheManager(cfg, DIRS["cache_dir"])
