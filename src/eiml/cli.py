from __future__ import annotations

from typing import Optional, Sequence
import argparse
import pprint

from .config import load_config_yaml, params_from_config
from .io_structures import read_structure_from_cfg
from .descriptor import compute_descriptor
from .save import save_features


def _parse_cli(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="eiml",
        description="Compute SOAP/EIML descriptors from a YAML config.",
    )
    p.add_argument("--config", type=str, required=True, help="Path to YAML config")
    p.add_argument(
        "--debug",
        action="store_true",
        help="Print parsed configuration and parameters (no descriptor values)",
    )
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = _parse_cli(argv)

    cfg = load_config_yaml(args.config)

    # params_from_config returns:
    # mode, structure_cfg, soap_params, saft_params, eiml_params, pool, output_file
    mode, structure_cfg, soap_params, saft_params, eiml_params, pool, output_file = params_from_config(cfg)

    atoms_or_traj = read_structure_from_cfg(structure_cfg)

    out_cfg = cfg.get("output", {}) or {}
    split_identity = bool(out_cfg.get("split_identity", False))

    # If user requests split_identity in SOAP mode, ignore (but do not crash)
    if split_identity and mode == "soap":
        print("[WARN] output.split_identity is ignored when mode='soap'. Writing a normal .npy instead.")
        split_identity = False

    if args.debug:
        print("\n=== Raw config (resolved) ===")
        pprint.pprint(cfg)
        print("\n=== Parsed parameters ===")
        print("mode:", mode)
        print("structure_cfg:", structure_cfg)
        print("soap_params:", soap_params)
        print("saft_params:", saft_params)
        print("eiml_params:", eiml_params)
        print("pool:", pool)
        print("output_file:", output_file)
        print("split_identity:", split_identity)
        print()

    # Compute descriptor
    if split_identity:
        # Return (X_geom, theta_id). In your current direction (no SAFT focus),
        # theta may be None; save_features can still store it if you keep that behavior.
        X_or_pair = compute_descriptor(
            atoms_or_traj,
            mode=mode,
            soap_params=soap_params,
            eiml_params=eiml_params,
            saft_params=saft_params,
            pool=pool,
            append_identity=False,
            return_identity=True,
        )
    else:
        # Return X only (and append identity if present + requested)
        X_or_pair = compute_descriptor(
            atoms_or_traj,
            mode=mode,
            soap_params=soap_params,
            eiml_params=eiml_params,
            saft_params=saft_params,
            pool=pool,
            append_identity=True,
            return_identity=False,
        )

    # Save
    save_features(
        mode=mode,
        output_file=output_file,
        split_identity=split_identity,
        X_or_pair=X_or_pair,
    )


if __name__ == "__main__":
    main()
