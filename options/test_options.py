"""This script contains the test options for Deep3DFaceRecon_pytorch"""

from .base_options import BaseOptions


class TestOptions(BaseOptions):
    """This class includes test options.

    It also includes shared options defined in BaseOptions.
    """

    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)  # define shared options
        parser.add_argument(
            "--phase",
            type=str,
            default="test",
            help="train, val, test, etc",
        )
        parser.add_argument(
            "--dataset_mode",
            type=str,
            default=None,
            help="chooses how datasets are loaded. [None | flist]",
        )
        parser.add_argument(
            "--img_folder", type=str, default="examples", help="folder for test images."
        )
        
        group = parser.add_mutually_exclusive_group(required=False)
        group.add_argument(
            "--visualize",
            "--vis",
            "--viz",
            dest="do_visualize",
            action="store_true",
            default=True,
            help="render/visualize the results.",
        )
        group.add_argument(
            "--no_visualize",
            "--no_vis",
            "--no_viz",
            dest="do_visualize",
            action="store_false",
            help="do not render/visualize the results.",
        )

        # Dropout and Batchnorm has different behavior during training and test.
        self.isTrain = False
        return parser
