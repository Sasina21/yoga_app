from enum import Enum

class ErrorCode(str, Enum):
    # Extract features 
    FILE_NOT_FOUND = "E101"
    IMREAD_FAILED = "E102"
    NO_LANDMARKS = "E103"

