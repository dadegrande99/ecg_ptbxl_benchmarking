class DataDirectoryError(Exception):
    """Exception raised when the data directory is not found."""
    def __init__(self, data_dir):
        super().__init__(f"The data directory '{data_dir}' does not exist.")


class ConfigFileError(Exception):
    """Exception raised when the configuration file is not found."""
    def __init__(self, config):
        super().__init__(f"The configuration file '{config}' does not exist.")
