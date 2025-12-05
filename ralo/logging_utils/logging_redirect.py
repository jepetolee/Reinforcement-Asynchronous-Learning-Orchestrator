"""
Logging redirect utility for capturing stdout/stderr to files while also printing to console.
"""
import sys
import os
from typing import Optional, TextIO


class Tee:
    """
    A file-like object that writes to both a file and stdout/stderr.
    """
    def __init__(self, file: TextIO, console: TextIO):
        self.file = file
        self.console = console

    def write(self, data: str) -> int:
        """Write data to both file and console."""
        self.file.write(data)
        self.file.flush()
        self.console.write(data)
        self.console.flush()
        return len(data)

    def flush(self) -> None:
        """Flush both file and console."""
        self.file.flush()
        self.console.flush()

    def close(self) -> None:
        """Close the file (but not console)."""
        self.file.close()


def setup_logging(log_file_path: str, also_log_stderr: bool = True) -> tuple[Optional[Tee], Optional[Tee]]:
    """
    Set up logging to redirect stdout (and optionally stderr) to a file while also printing to console.
    
    Args:
        log_file_path: Path to the log file
        also_log_stderr: If True, also redirect stderr to the same file
    
    Returns:
        Tuple of (stdout_tee, stderr_tee) objects that can be used to restore original streams
    """
    # Create directory if it doesn't exist
    log_dir = os.path.dirname(log_file_path)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)
    
    # Open log file in append mode
    log_file = open(log_file_path, 'a', encoding='utf-8')
    
    # Create Tee objects
    stdout_tee = Tee(log_file, sys.stdout)
    stderr_tee = None
    
    if also_log_stderr:
        # For stderr, we'll write to the same file but keep stderr for console
        stderr_file = open(log_file_path, 'a', encoding='utf-8')
        stderr_tee = Tee(stderr_file, sys.stderr)
        sys.stderr = stderr_tee
    
    # Redirect stdout
    sys.stdout = stdout_tee
    
    return stdout_tee, stderr_tee


def restore_logging(stdout_tee: Optional[Tee], stderr_tee: Optional[Tee]) -> None:
    """
    Restore original stdout/stderr streams.
    
    Args:
        stdout_tee: The Tee object that was used for stdout
        stderr_tee: The Tee object that was used for stderr
    """
    if stdout_tee:
        sys.stdout = stdout_tee.console
        stdout_tee.close()
    
    if stderr_tee:
        sys.stderr = stderr_tee.console
        stderr_tee.close()

