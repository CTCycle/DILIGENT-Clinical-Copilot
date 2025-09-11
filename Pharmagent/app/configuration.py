###############################################################################
class Configuration:
    def __init__(self) -> None:
        self.configuration = {
            "model": "gpt-3.5-turbo",
        }

    # -------------------------------------------------------------------------
    def get_configuration(self) -> dict[str, Any]:
        return self.configuration
