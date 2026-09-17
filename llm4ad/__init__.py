import os

from . import base

# Experiments can opt out of importing every optional method/task at startup.
# Default behavior and the GUI's dynamic discovery remain unchanged.
if os.environ.get('LLM4AD_MINIMAL_IMPORTS') != '1':
    from . import method
    from . import task
    from .tools import profiler
    from .tools import llm

__version__ = '1.0.0'
