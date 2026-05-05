class HVACDomainError(Exception):
    error_code: str
    http_status: int

class FileCorruptionError(HVACDomainError):
    error_code  = "FILE_CORRUPTION"
    http_status = 400

class OCRTimeoutError(HVACDomainError):
    error_code  = "OCR_TIMEOUT"
    http_status = 504

class UnreadableBlueprintError(HVACDomainError):
    error_code  = "UNREADABLE_BLUEPRINT"
    http_status = 422

class PayloadTooLargeError(HVACDomainError):
    error_code  = "PAYLOAD_TOO_LARGE"
    http_status = 413

class UnsupportedMediaTypeError(HVACDomainError):
    error_code  = "UNSUPPORTED_MEDIA_TYPE"
    http_status = 415
