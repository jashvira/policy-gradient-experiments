#!/usr/bin/env python3
"""
Monkey patch for vLLM Python 3.13 compatibility.
Fixes OSError: 'source code not available' in get_attr_docs.
"""

def patch_get_attr_docs():
    """Patch vLLM's get_attr_docs to handle Python 3.13 inspect failures."""
    import vllm.config
    from dataclasses import fields

    original_get_attr_docs = vllm.config.get_attr_docs

    def safe_get_attr_docs(cls):
        try:
            return original_get_attr_docs(cls)
        except OSError:
            # Return empty docs for all fields when source code is not available
            from dataclasses import is_dataclass
            if is_dataclass(cls):
                return {field.name: "" for field in fields(cls)}
            return {}

    vllm.config.get_attr_docs = safe_get_attr_docs

if __name__ == "__main__":
    patch_get_attr_docs()

    # Import and run vLLM server after patching
    from verifiers.inference.vllm_server import main
    main()
