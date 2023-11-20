import joblib
from pydantic_settings import BaseSettings


class Env(BaseSettings):
    BIONANO_IMAGES_DIR: str
    DEEPOM_MODEL_FILE: str = 'src/deepom/deepom.pt'
    CACHE_DIR: str = 'out/cache'


ENV = Env()
joblib_memory = joblib.Memory(ENV.CACHE_DIR, verbose=0)
