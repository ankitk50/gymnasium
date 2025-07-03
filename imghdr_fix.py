# Temporary fix for imghdr module compatibility with Python 3.13
"""
Minimal imghdr replacement for TensorBoard compatibility.
This module provides basic image format detection.
"""

def what(filename=None, h=None):
    """Detect image format from file or bytes."""
    import os
    
    if filename and os.path.exists(filename):
        with open(filename, 'rb') as f:
            header = f.read(32)
    elif h:
        header = h[:32] if len(h) >= 32 else h
    else:
        return None
    
    # Basic image format detection
    if header.startswith(b'\xff\xd8\xff'):
        return 'jpeg'
    elif header.startswith(b'\x89PNG\r\n\x1a\n'):
        return 'png'
    elif header.startswith(b'GIF87a') or header.startswith(b'GIF89a'):
        return 'gif'
    elif header.startswith(b'RIFF') and header[8:12] == b'WEBP':
        return 'webp'
    elif header.startswith(b'BM'):
        return 'bmp'
    else:
        return None

# Add this module to sys.modules as imghdr
import sys
sys.modules['imghdr'] = sys.modules[__name__]
