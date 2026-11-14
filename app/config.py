from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    GOOGLE_API_KEY: str = ""
    CLASSIFIER_MODEL: str = "google:gemini-3.5-flash-lite"
    AGENT_MODEL: str = "google:gemini-3.8-flash"
    GOOGLE_CLIENT_ID: str = ""
    GOOGLE_CLIENT_SECRET: str = ""
    GOOGLE_REFRESH_TOKEN: str = ""
    DATABASE_URL: str = ""
    POLLER_INTERVAL_SECONDS: int = 30
    DASHBOARD_PASSWORD: str = "changeme"


settings = Settings()
