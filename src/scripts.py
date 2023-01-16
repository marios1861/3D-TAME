import argparse
import val
import train
import other_methods_vit as other
import transformer_explainability_val as hila


def parse_train(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--epoch",
        type=int,
        default=None,
        help="Epoch to load, defaults to latest epoch saved. -1 to restart training",
    )
    parser.set_defaults(func=train.main)


def parse_val(parser: argparse.ArgumentParser):
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--with-val", action="store_true", help="test with val dataset")
    group.add_argument(
        "--epoch", type=int, default=None, help="Chosen epoch from validation"
    )
    parser.set_defaults(func=val.main)


def parse_other(parser: argparse.ArgumentParser):
    parser.add_argument("--with-val", action="store_true", help="test with val dataset")
    subparsers = parser.add_subparsers()

    grad_parser = subparsers.add_parser(
        "grad",
        description="Evaluation script for methods included in pytorch_grad_cam library",
    )
    grad_parser.add_argument(
        "--method",
        type=str,
        default="GradCam",
        help="explainability method",
        choices=[
            "gradcam",
            "scorecam",
            "gradcam++",
            "ablationcam",
            "xgradcam",
            "eigencam",
            "eigengradcam",
            "layercam",
            "fullgrad",
        ],
    )
    grad_parser.set_defaults(func=other.main)

    hila_parser = subparsers.add_parser(
        "hila",
        description="Evaluation script for method developed in Transformer Explainability",
    )
    hila_parser.set_defaults(func=hila.main)


def main():
    parser = argparse.ArgumentParser(prog="TAME")
    parser.add_argument(
        "--cfg", type=str, default="default.yaml", help="config script name (not path)"
    )
    subparsers = parser.add_subparsers()

    # create the parser for the train command
    train_parser = subparsers.add_parser("train", description="Train script for TAME")
    parse_train(train_parser)

    # create the parser for the val command
    val_parser = subparsers.add_parser("val", description="Eval script for TAME")
    parse_val(val_parser)

    # other methods
    other_parser = subparsers.add_parser(
        "other-val", description="Eval script for other methods"
    )
    parse_other(other_parser)

    args = parser.parse_args()
    args.func(vars(args))
