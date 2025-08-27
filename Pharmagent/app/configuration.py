###############################################################################
class Configuration:
    def __init__(self):
        self.configuration = {
            "model": "gpt-3.5-turbo",
        }

    # -------------------------------------------------------------------------
    def get_configuration(self):
        return self.configuration
