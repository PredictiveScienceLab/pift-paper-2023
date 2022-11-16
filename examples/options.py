"""Some standard options."""


__all__ = ["make_standard_option_parser"]


from optparse import OptionParser

def make_standard_option_parser() -> OptionParser:
    parser = OptionParser()

    # MCMC parameters
    parser.add_option(
        "--mini-batch",
        dest="mini_batch",
        help="the mini batch used for energy conserving subsampling",
        type="int",
        default=1000
    )
    parser.add_option(
        "--num-warmup",
        dest="num_warmup",
        help="number of warmup steps for MCMC",
        type="int",
        default=1000
    )
    parser.add_option(
        "--thinning",
        dest="thinning",
        help="keep every `thinning` samples from MCMC",
        type="int",
        default=1
    )
    parser.add_option(
        "--num-samples",
        dest="num_samples",
        help="total number of MCMC samplers",
        type="int",
        default=100
    )

    # Function parameterization paramaters
    parser.add_option(
        "--num-terms",
        dest="num_terms",
        help="number of Fourier features",
        type="int",
        default=4
    )
    
    return parser
