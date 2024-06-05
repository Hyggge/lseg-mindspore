from modules.lseg_module import LSegModule
from utils import do_training, get_default_argument_parser

if __name__ == "__main__":
    parser = LSegModule.add_model_specific_args(get_default_argument_parser())
    args = parser.parse_args()
    for k, v in vars(args).items():
        print(f"{k}: {v}")
    do_training(args, LSegModule)
