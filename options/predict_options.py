from .base_options import BaseOptions


class PredictOptions(BaseOptions):
    """This class includes predict options.

    It also includes shared options defined in BaseOptions.
    """

    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)  # define shared options
        parser.add_argument('--results_dir', type=str, default='./results/', help='saves results here.')
        parser.add_argument('--aspect_ratio', type=float, default=1.0, help='aspect ratio of result images')
        parser.add_argument('--phase', type=str, default='predict', help='train, val, test, predict, etc')
        # rewrite devalue values
        parser.set_defaults(model='predict')

        self.isTrain = False
        return parser
