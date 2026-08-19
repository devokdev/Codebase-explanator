"""Backend package for the codebase understanding system."""
import sys
# Prevent broken Windows environment packages from crashing transformers / sentence_transformers import chain
sys.modules.setdefault("torchvision", None)
sys.modules.setdefault("tensorflow", None)
