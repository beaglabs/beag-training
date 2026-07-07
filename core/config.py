from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    deepseek_api_key: str = ""
    deepseek_model: str = "deepseek-chat"

    class Config:
        env_file = None


settings = Settings()
