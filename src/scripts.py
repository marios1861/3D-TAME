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
        help="epoch to load, set to -1 to restart training, omit to restart on last saved epoch",
    )
    parser.set_defaults(func=train.main)


def parse_val(parser: argparse.ArgumentParser):
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--with-val", action="store_true", help="test with validation dataset"
    )
    group.add_argument(
        "--epoch", type=int, default=None, help="epoch of TAME to evaluate on test set"
    )
    parser.set_defaults(func=val.main)


def parse_other(
    parser: argparse.ArgumentParser, general_parser: argparse.ArgumentParser
):
    subparsers = parser.add_subparsers()

    grad_parser = subparsers.add_parser(
        "grad",
        description="Evaluation script for methods included in pytorch_grad_cam library",
        parents=[general_parser],
    )
    grad_parser.add_argument(
        "--method",
        type=str,
        help="explainability method to evaluate; omit to evaluate with every method",
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
        parents=[general_parser],
    )
    hila_parser.set_defaults(func=hila.main)


def main():
    general_parser = argparse.ArgumentParser(add_help=False)
    general_parser.add_argument(
        "--cfg", type=str, default="default", help="config script to use (not path)"
    )
    parser = argparse.ArgumentParser(
        prog="TAME",
        description="Trainable Attention Mechanism for Explanations",
    )
    subparsers = parser.add_subparsers(title="subcommands")

    # create the parser for the train command
    train_parser = subparsers.add_parser(
        "train",
        help="training subcommand",
        description="Train script for TAME",
        parents=[general_parser],
    )
    parse_train(train_parser)

    # create the parser for the val command
    val_parser = subparsers.add_parser(
        "val",
        help="tame evaluation subcommand",
        description="Eval script for TAME",
        parents=[general_parser],
    )
    parse_val(val_parser)

    # other methods
    other_parser = subparsers.add_parser(
        "other-val",
        help="evaluation for other methods subcommand",
        description="Eval script for other methods",
        parents=[general_parser],
    )
    parse_other(other_parser, general_parser)

    args = parser.parse_args()
    args.func(vars(args))
