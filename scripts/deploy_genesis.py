#!/usr/bin/env python3
import argparse
import os
from core.controller_new import HomieGenesisControllerNew
import cfg.robot_config as cfg

def main():
    parser = argparse.ArgumentParser(description="Genesis deploy for Homie locomotion policy")
    parser.add_argument("--policy", type=str, default=cfg.DEFAULT_POLICY_PATH, help="Path to TorchScript Homie policy (.pt)")
    parser.add_argument("--robot", type=str, default=cfg.DEFAULT_ROBOT_XML, help="Path to G1 MJCF/XML")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--no-render", action="store_true")
    parser.add_argument("--dt", type=float, default=0.02)
    parser.add_argument("--decimation", type=int, default=1)
    parser.add_argument("--duration", type=float, default=300.0)
    parser.add_argument("--high-policy", type=str, default=cfg.DEFAULT_HIGH_POLICY_PATH, help="Path to high-level policy")
    args = parser.parse_args()

    hl_path = args.high_policy if os.path.exists(args.high_policy) else None

    ctrl = HomieGenesisControllerNew(
        homie_policy_path=args.policy,
        robot_xml_path=args.robot,
        device=args.device,
        render=not args.no_render,
        dt=args.dt,
        control_decimation=args.decimation,
        high_level_policy_path=hl_path,
    )
    ctrl.run(duration_sec=args.duration)

if __name__ == "__main__":
    main()