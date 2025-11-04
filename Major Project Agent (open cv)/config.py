"""
Configuration system for the Multi-Agent Navigation System
Updated for OpenCV DNN YOLO implementation
"""

import json
import os
import tempfile
from typing import Dict, Any
from dataclasses import dataclass, asdict

# Use project root (file location) as base so relative runs from anywhere still find config
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CONFIG_PATH = os.path.join(BASE_DIR, "config.json")


# =======================
#  PERCEPTION CONFIG
# =======================
@dataclass
class PerceptionConfig:
    """Configuration for Perception Agent with OpenCV DNN"""
    proximity_threshold: float = 2.0
    confidence_threshold: float = 0.5
    nms_threshold: float = 0.4
    camera_index: int = 0
    frame_width: int = 640
    frame_height: int = 480
    trapezoid: Dict[str, int] = None
    
    # OpenCV DNN YOLO configuration
    model_cfg: str = "models/yolov4-tiny.cfg"
    model_weights: str = "models/yolov4-tiny.weights"
    class_names: str = "models/coco.names"
    input_size: int = 416
    scale: float = 1/255.0
    mean: tuple = (0, 0, 0)
    swap_rb: bool = True

    def __post_init__(self):
        if self.trapezoid is None:
            self.trapezoid = {
                'top_height': 100,
                'bottom_height': 250,
                'width': 400
            }


# =======================
#  NAVIGATION CONFIG
# =======================
@dataclass
class NavigationConfig:
    """Configuration for Navigation Agent"""
    safe_distance: float = 2.0
    warning_distance: float = 3.0
    path_width: float = 1.0
    update_frequency: float = 0.5
    risk_thresholds: Dict[str, float] = None

    def __post_init__(self):
        if self.risk_thresholds is None:
            self.risk_thresholds = {
                'low': 3.0,
                'medium': 2.0,
                'high': 1.0,
                'critical': 0.3
            }


# =======================
#  COMMUNICATION CONFIG
# =======================
@dataclass
class CommunicationConfig:
    """Configuration for Communication Agent"""
    tts_enabled: bool = True
    visual_enabled: bool = True
    speech_rate: int = 160
    speech_volume: float = 0.8
    delay_seconds: int = 4
    display: Dict[str, Any] = None

    def __post_init__(self):
        if self.display is None:
            self.display = {
                'width': 640,
                'height': 480,
                'font_scale': 0.8,
                'font_thickness': 2,
                'alert_duration': 4.0
            }


# =======================
#  SYSTEM CONFIG
# =======================
@dataclass
class SystemConfig:
    """Main system configuration"""
    perception: PerceptionConfig
    navigation: NavigationConfig
    communication: CommunicationConfig

    # Global settings
    debug_mode: bool = False
    log_level: str = "INFO"

    # Display settings
    show_fps: bool = True
    show_depth_map: bool = True
    window_names: Dict[str, str] = None

    def __post_init__(self):
        if self.window_names is None:
            self.window_names = {
                'main': "Multi-Agent Navigation System (OpenCV DNN)",
                'depth': "Depth Map"
            }


# =======================
#  CONFIG MANAGER
# =======================
class ConfigManager:
    """Manages system configuration"""

    def __init__(self, config_path: str = None):
        self.config_path = config_path or DEFAULT_CONFIG_PATH
        self.config: SystemConfig = self._load_config()

    def _load_config(self) -> SystemConfig:
        """Load configuration from file or create default"""
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                # Create config objects from loaded data
                perception_config = PerceptionConfig(**data.get('perception', {}))
                navigation_config = NavigationConfig(**data.get('navigation', {}))
                communication_config = CommunicationConfig(**data.get('communication', {}))

                # Create system config
                system_data = {k: v for k, v in data.items()
                               if k not in ['perception', 'navigation', 'communication']}

                config = SystemConfig(
                    perception=perception_config,
                    navigation=navigation_config,
                    communication=communication_config,
                    **system_data
                )

                print(f"[config] Loaded configuration from {self.config_path}")
                return config

            except Exception as e:
                print(f"[config] Error loading config ({e}). Using defaults.")

        # Create default configuration if file not found or failed
        config = SystemConfig(
            perception=PerceptionConfig(),
            navigation=NavigationConfig(),
            communication=CommunicationConfig()
        )

        # Save default configuration
        try:
            self.save_config(config)
        except Exception:
            pass
        return config

    def save_config(self, config: SystemConfig = None):
        """Save configuration to file (atomic write)"""
        if config is None:
            config = self.config

        try:
            config_dict = {
                'perception': asdict(config.perception),
                'navigation': asdict(config.navigation),
                'communication': asdict(config.communication),
                'debug_mode': config.debug_mode,
                'log_level': config.log_level,
                'show_fps': config.show_fps,
                'show_depth_map': config.show_depth_map,
                'window_names': config.window_names
            }

            # Atomic write: write to temp file then replace
            dir_name = os.path.dirname(self.config_path) or '.'
            with tempfile.NamedTemporaryFile('w', delete=False, dir=dir_name, encoding='utf-8') as tmp:
                json.dump(config_dict, tmp, indent=4)
                temp_name = tmp.name

            os.replace(temp_name, self.config_path)
            print(f"[config] Configuration saved to {self.config_path}")

        except Exception as e:
            print(f"[config] Error saving config: {e}")

    # --- Getters ---
    def get_perception_config(self) -> Dict[str, Any]:
        return asdict(self.config.perception)

    def get_navigation_config(self) -> Dict[str, Any]:
        return asdict(self.config.navigation)

    def get_communication_config(self) -> Dict[str, Any]:
        return asdict(self.config.communication)

    # --- Updaters ---
    def update_perception_config(self, **kwargs):
        for key, value in kwargs.items():
            if hasattr(self.config.perception, key):
                setattr(self.config.perception, key, value)
        self.save_config()

    def update_navigation_config(self, **kwargs):
        for key, value in kwargs.items():
            if hasattr(self.config.navigation, key):
                setattr(self.config.navigation, key, value)
        self.save_config()

    def update_communication_config(self, **kwargs):
        for key, value in kwargs.items():
            if hasattr(self.config.communication, key):
                setattr(self.config.communication, key, value)
        self.save_config()

    def update_system_config(self, **kwargs):
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
        self.save_config()

    def reset_to_defaults(self):
        """Reset configuration to defaults"""
        self.config = SystemConfig(
            perception=PerceptionConfig(),
            navigation=NavigationConfig(),
            communication=CommunicationConfig()
        )
        self.save_config()
        print("[config] Configuration reset to defaults")


# Global config manager instance
config_manager = ConfigManager()