import os
import time


def uuid7() -> str:
    """Generate a UUIDv7 string: 48-bit millisecond timestamp + random bits."""
    unix_ms = int(time.time() * 1000)
    ts_bytes = unix_ms.to_bytes(6, byteorder="big")
    rand = bytearray(os.urandom(10))

    # Version 7 in top nibble of byte 6, variant bits (10xx) in byte 8.
    rand[0] = (0x70 | (rand[0] & 0x0F))
    rand[2] = (0x80 | (rand[2] & 0x3F))

    raw = ts_bytes + bytes(rand)
    hex_str = raw.hex()
    return (
        f"{hex_str[0:8]}-{hex_str[8:12]}-{hex_str[12:16]}-"
        f"{hex_str[16:20]}-{hex_str[20:32]}"
    )
