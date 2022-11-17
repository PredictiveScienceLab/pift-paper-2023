"""Some standard options."""


__all__ = [
    "make_standard_option_parser",
    "add_nr_options",
    "add_sgld_options"
]


from argparse import ArgumentParser

def make_standard_option_parser(**kwargs) -> ArgumentParser:
    parser = ArgumentParser(**kwargs)

    # MCMC parameters
    parser.add_argument(
        "--mini-batch",
        dest="mini_batch",
        help="the mini batch used for energy conserving subsampling",
        type=int,
        default=1000
    )
    parser.add_argument(
        "--num-warmup",
        dest="num_warmup",
        help="number of warmup steps for MCMC",
        type=int,
        default=1000
    )
    parser.add_argument(
        "--thinning",
        dest="thinning",
        help="keep every `thinning` samples from MCMC",
        type=int,
        default=1
    )
    parser.add_argument(
        "--num-samples",
        dest="num_samples",
        help="total number of MCMC samplers",
        type=int,
        default=100
    )
    parser.add_argument(
        "--progress-bar",
        dest="progress_bar",
        help="show the progress bar during MCMC sampling",
        action="store_true",
        default=False
    )

    # Function parameterization paramaters
    parser.add_argument(
        "--num-terms",
        dest="num_terms",
        help="number of Fourier features",
        type=int,
        default=4
    )

    return parser


def add_nr_options(parser):
    parser.add_argument(
        "--nr-alpha",
        dest="nr_alpha",
        help="the learning rate for NR",
        type=float,
        default=1.0
    )
    parser.add_argument(
        "--nr-maxit",
        dest="nr_maxit",
        help="the maximum number of iterations for NR",
        type=int,
        default=100
    )
    parser.add_argument(
        "--nr-tol",
        dest="nr_tol",
        help="the tolerance for NR",
        type=float,
        default=1e-2
    )


def add_sgld_options(parser):
    parser.add_argument(
        "--sgld-alpha",
        dest="sgld_alpha",
        help="the learning rate coefficient for SGLD",
        type=float,
        default=0.1
    )
    parser.add_argument(
        "--sgld-beta",
        dest="sgld_beta",
        help="the learning rate offset for SGLD",
        type=float,
        default=0.0
    )
    parser.add_argument(
        "--sgld-gamma",
        dest="sgld_gamma",
        help="the learning rate dedcay coefficient for SGLD",
        type=float,
        default=0.51
    )
    parser.add_argument(
        "--sgld-maxit",
        dest="sgld_maxit",
        help="the maximum number of iterations for SGLD",
        type=int,
        default=1000
    )
    parser.add_argument(
        "--sgld-fix-it",
        dest="sgld_fix_it",
        help="the iterations after which the learning rate is fixed in SGLD",
        type=int,
        default=500
    )
