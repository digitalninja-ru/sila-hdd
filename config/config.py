from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_ignore_empty=True)

    ARCHIVE_PATH: str = ''
    ZIP_DATA_PATH: str = ''
    DATASET_PATH: str = ''

settings = Settings()
