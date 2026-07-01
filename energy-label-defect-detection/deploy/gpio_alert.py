"""
GPIO 报警模块.
"""
import yaml
from utils.logger import get_logger

logger = get_logger(__name__)


class GPIOAlert:
    def __init__(self, config_path: str = "config.yaml"):
        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f)
        gpio_cfg = cfg.get("gpio", {})
        self.enabled = gpio_cfg.get("enabled", False)
        self.pin_led = gpio_cfg.get("pin_led", 17)
        self.pin_buzzer = gpio_cfg.get("pin_buzzer", 27)
        self._gpio = None
        if self.enabled:
            self._init_gpio()

    def _init_gpio(self):
        try:
            import RPi.GPIO as GPIO
            GPIO.setmode(GPIO.BCM)
            GPIO.setup(self.pin_led, GPIO.OUT)
            GPIO.setup(self.pin_buzzer, GPIO.OUT)
            self._gpio = GPIO
            logger.info("GPIO initialized (RPi.GPIO).")
        except ImportError:
            try:
                from pyA20.gpio import gpio, port
                gpio.init()
                gpio.setcfg(self.pin_led, gpio.OUTPUT)
                gpio.setcfg(self.pin_buzzer, gpio.OUTPUT)
                self._gpio = gpio
                logger.info("GPIO initialized (pyA20).")
            except ImportError:
                logger.warning("No GPIO library found. GPIO disabled.")
                self.enabled = False

    def trigger(self, duration_ms: int = 500):
        if not self.enabled or self._gpio is None:
            return
        try:
            if hasattr(self._gpio, 'output'):
                self._gpio.output(self.pin_led, 1)
                self._gpio.output(self.pin_buzzer, 1)
                import time
                time.sleep(duration_ms / 1000.0)
                self._gpio.output(self.pin_led, 0)
                self._gpio.output(self.pin_buzzer, 0)
        except Exception as e:
            logger.error(f"GPIO trigger failed: {e}")

    def cleanup(self):
        if self._gpio is not None:
            try:
                if hasattr(self._gpio, 'cleanup'):
                    self._gpio.cleanup()
            except Exception:
                pass
