"""Early process setup — import and call init() FIRST at every entry point,
before importing app modules or creating any network clients.

Handles two Windows/environment papercuts:
  1. TLS trust: routes Python's SSL through the OS certificate store so HTTPS
     works behind TLS-inspecting antivirus/proxies (e.g. Avast) whose root CA
     is in the Windows store but not in Python's bundled `certifi`.
  2. Console encoding: forces UTF-8 so the emoji status prints don't crash on
     legacy Windows consoles (cp1252).
"""
import sys

_done = False


def init():
    global _done
    if _done:
        return
    _done = True

    # 1. Use the OS trust store (must run before SSL contexts are created)
    try:
        import truststore
        truststore.inject_into_ssl()
    except Exception as e:  # truststore missing / unsupported — keep going
        print(f"⚠️ truststore not active ({e}); using default certifi bundle.")

    # 2. Force UTF-8 output
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except (AttributeError, ValueError):
            pass
