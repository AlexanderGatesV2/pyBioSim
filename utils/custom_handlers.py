import os
import re
import logging
from logging.handlers import RotatingFileHandler

class CustomRotatingFileHandler(RotatingFileHandler):
    """
    Custom RotatingFileHandler that uses name(N).log format for rotated files
    instead of the default name.log.N format.
    """
    def rotation_filename(self, default_name):
        """
        Modify the rotated filename to use name(N).log format.
        
        Args:
            default_name: The default filename that would be used (name.log.N)
            
        Returns:
            Modified filename in the format name(N).log
        """
        # Extract the base name and number from the default rotation pattern
        match = re.match(r'(.+)\.log\.(\d+)$', default_name)
        if match:
            base_name = match.group(1)
            number = match.group(2)
            # Create the new filename format
            return f"{base_name}({number}).log"
        return default_name
