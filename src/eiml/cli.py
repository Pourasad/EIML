from __future__ import annotations

from typing import Optional, Sequence
import argparse

from .config import load_config_yaml, params_from_config
from .io_structures import read_structure_from_cfg
from .descriptor import compute_descriptor
from .save import save_features


def _parse_cli(argv: Optional[Sequence[str]] = None):
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default=None, help="Path to YAML config")
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None):
    args = _parse_cli(argv)

    if args.config is None:
        raise SystemExit("Provide --config config.yaml")

    cfg = load_config_yaml(args.config)

    # NOTE: params_from_config must return:
    # mode, structure_cfg, soap_params, saft_params, eiml_params, pool, output_file
    mode, structure_cfg, soap_params, saft_params, eiml_params, pool, output_file = params_from_config(cfg)
    atoms = read_structure_from_cfg(structure_cfg)

    out = cfg.get("output", {})
    split_identity_requested = bool(out.get("split_identity", False))

    # If user requests split_identity in SOAP mode, ignore (but do not crash)
    if split_identity_requested and mode == "soap":
        print("[WARN] output.split_identity is ignored when mode='soap'. Writing a normal .npy instead.")
        split_identity_requested = False

    # We allow it in mode eiml, but it will only split if saft_params exists.
    if split_identity_requested and mode == "eiml":
        X = compute_descriptor(
            atoms=atoms,
            mode=mode,
            soap_params=soap_params,
            eiml_params=eiml_params,
            saft_params=saft_params,  # optional identity
            pool=pool,
            append_identity=False,
            return_identity=True,
        )
    else:
        X = compute_descriptor(
            atoms=atoms,
            mode=mode,
            soap_params=soap_params,
            eiml_params=eiml_params,
            saft_params=saft_params,
            pool=pool,
            append_identity=True,
            return_identity=False,
        )

    save_features(
        mode=mode,
        output_file=output_file,
        split_identity_requested=split_identity_requested,
        X_or_pair=X,
    )


if __name__ == "__main__":
    main()
