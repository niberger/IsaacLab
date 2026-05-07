# Copyright (c) 2024-2025, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#

import os
import sys


def bootstrap_kernel():
    # Isaac Lab path
    isaaclab_path = os.path.dirname(os.path.abspath(os.path.realpath(__file__)))

    # bootstrap kernel via Isaac Sim
    import isaacsim

    # log info
    import carb
    carb.log_info(f"Isaac Lab path: {isaaclab_path}")

def expose_api():
    try:
        # get 'isaaclab/app' folder path
        isaaclab_path = os.path.dirname(os.path.abspath(os.path.realpath(__file__)))
        path = os.path.join(isaaclab_path, "source", "isaaclab", "isaaclab", "app")
        if os.path.exists(path):
            # register path
            sys.path.insert(0, path)
            
            from settings_manager import SettingsManager, get_settings_manager, initialize_carb_settings

            sys.modules["isaaclab.app"] = type(sys)("isaaclab.app")
            sys.modules["isaaclab.app.settings_manager"] = type(sys)("isaaclab.app.settings_manager")
            sys.modules["isaaclab.app.settings_manager.SettingsManager"] = SettingsManager
            sys.modules["isaaclab.app.settings_manager.get_settings_manager"] = get_settings_manager
            sys.modules["isaaclab.app.settings_manager.initialize_carb_settings"] = initialize_carb_settings

            from app_launcher import AppLauncher
            
            sys.modules["isaaclab.app.AppLauncher"] = AppLauncher
        else:
            print(f"PYTHONPATH: path doesn't exist ({path})")
    except ImportError as e:
        print(f"Unable to expose 'isaaclab.app' API: {e}")

def main():
    args = sys.argv[1:]
    raise NotImplementedError(str(args))


bootstrap_kernel()
expose_api()
