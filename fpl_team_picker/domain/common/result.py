"""Result types for frontend-agnostic error handling."""

from enum import Enum
from typing import Dict, Generic, Optional, TypeVar

from pydantic import BaseModel, Field

T = TypeVar("T")


class ErrorType(str, Enum):
    """Standard error types for consistent handling across frontends."""

    VALIDATION_ERROR = "validation_error"
    DATA_NOT_FOUND = "data_not_found"
    DATA_ACCESS_ERROR = "data_access_error"
    EXTERNAL_API_ERROR = "external_api_error"
    BUSINESS_RULE_VIOLATION = "business_rule_violation"
    CONFIGURATION_ERROR = "configuration_error"
    CALCULATION_ERROR = "calculation_error"
    SYSTEM_ERROR = "system_error"


class DomainError(BaseModel):
    """Structured error information for frontend consumption."""

    error_type: ErrorType = Field(..., description="Standardized error type")
    message: str = Field(..., min_length=1, description="Human-readable error message")
    details: Optional[Dict] = Field(None, description="Additional error context")
    error_code: Optional[str] = Field(
        None, description="Specific error code for handling"
    )
    field_errors: Optional[Dict[str, str]] = Field(
        None, description="Field-specific validation errors"
    )

    @classmethod
    def validation_error(
        cls,
        message: str,
        field_errors: Optional[Dict[str, str]] = None,
        details: Optional[Dict] = None,
    ) -> "DomainError":
        """Create a validation error."""
        return cls(
            error_type=ErrorType.VALIDATION_ERROR,
            message=message,
            field_errors=field_errors,
            details=details,
        )

    @classmethod
    def data_not_found(
        cls, message: str, details: Optional[Dict] = None
    ) -> "DomainError":
        """Create a data not found error."""
        return cls(
            error_type=ErrorType.DATA_NOT_FOUND, message=message, details=details
        )

    @classmethod
    def business_rule_violation(
        cls, message: str, details: Optional[Dict] = None
    ) -> "DomainError":
        """Create a business rule violation error."""
        return cls(
            error_type=ErrorType.BUSINESS_RULE_VIOLATION,
            message=message,
            details=details,
        )

    @classmethod
    def external_api_error(
        cls, message: str, details: Optional[Dict] = None
    ) -> "DomainError":
        """Create an external API error."""
        return cls(
            error_type=ErrorType.EXTERNAL_API_ERROR, message=message, details=details
        )


class Result(Generic[T]):
    """
    Result type for frontend-agnostic error handling.

    Allows domain operations to return either success values or structured errors
    without throwing exceptions that frontend adapters need to catch.
    """

    def __init__(
        self,
        value: Optional[T] = None,
        error: Optional[DomainError] = None,
        _allow_none: bool = False,
    ):
        if value is not None and error is not None:
            raise ValueError("Result cannot have both value and error")
        if not _allow_none and value is None and error is None:
            raise ValueError("Result must have either value or error")

        self._value = value
        self._error = error

    @property
    def value(self) -> T:
        """Get the success value. Raises error if result is failure."""
        if self._error is not None:
            raise ValueError(
                f"Cannot access value on failed result: {self._error.message}"
            )
        return self._value

    @property
    def error(self) -> DomainError:
        """Get the error. Raises error if result is success."""
        if self._value is not None:
            raise ValueError("Cannot access error on successful result")
        return self._error

    @property
    def is_success(self) -> bool:
        """Check if result represents success."""
        return self._error is None

    @property
    def is_failure(self) -> bool:
        """Check if result represents failure."""
        return self._error is not None

    @classmethod
    def success(cls, value: T) -> "Result[T]":
        """Create a successful result."""
        return cls(value=value, _allow_none=True)

    @classmethod
    def failure(cls, error: DomainError) -> "Result[T]":
        """Create a failed result."""
        return cls(error=error)

    def map(self, func) -> "Result":
        """Transform the value if successful, otherwise return the error."""
        if self.is_success:
            try:
                return Result.success(func(self.value))
            except Exception as e:
                return Result.failure(
                    DomainError(
                        error_type=ErrorType.CALCULATION_ERROR,
                        message=f"Transformation failed: {str(e)}",
                    )
                )
        else:
            return Result.failure(self.error)

    def flat_map(self, func) -> "Result":
        """Apply a function that returns a Result, flattening the result."""
        if self.is_success:
            try:
                return func(self.value)
            except Exception as e:
                return Result.failure(
                    DomainError(
                        error_type=ErrorType.CALCULATION_ERROR,
                        message=f"Operation failed: {str(e)}",
                    )
                )
        else:
            return Result.failure(self.error)
