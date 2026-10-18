"""Status LEDs on GPIO (QNX rpi_gpio). Falls back to printing on a laptop."""
from __future__ import annotations


class Lights:
    def __init__(self, pins: dict | None = None):
        self.pins = pins or {"green": 17, "yellow": 27, "red": 22}  # BCM numbering
        self.last = None
        try:
            import rpi_gpio as GPIO  # QNX Raspberry Pi GPIO module
            GPIO.setmode(GPIO.BCM)
            for p in self.pins.values():
                GPIO.setup(p, GPIO.OUT)
                GPIO.output(p, GPIO.LOW)
            self.gpio = GPIO
        except Exception as e:  # not on the Pi
            print(f"[lights] GPIO unavailable ({e}); printing state instead")
            self.gpio = None

    def show(self, state: str) -> None:
        if state == self.last:
            return
        self.last = state
        if self.gpio:
            for name, p in self.pins.items():
                self.gpio.output(p, self.gpio.HIGH if name == state else self.gpio.LOW)
        print(f"[lights] {state.upper()}", flush=True)

    def close(self) -> None:
        if self.gpio:
            for p in self.pins.values():
                self.gpio.output(p, self.gpio.LOW)
            self.gpio.cleanup()
