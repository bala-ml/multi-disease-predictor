from pydantic_settings import BaseSettings

class Settings(BaseSettings):

    log_path: str
    diabetes_model_path: str
    cardio_risk_model_path: str
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "allow"
