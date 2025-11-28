"""
This script runs `experiment_unittests.run_tests()` with an alternative `RADAR_ID` environment variable.
"""

# Do not add any Borealis-related imports here, as they will be cached and will not allow proper mocking of `RADAR_ID`
import argparse
import os


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "site_id",
        choices=["sas", "pgr", "inv", "rkn", "cly", "lab"],
        help="Site ID of site to test experiments as.",
    )
    parser.add_argument(
        "--experiments",
        required=False,
        nargs="+",
        default=None,
        help="Only run the experiments specified after this option. "
        "Experiments specified must exist within the top-level Borealis experiments directory.",
    )
    parser.add_argument(
        "--kwargs",
        required=False,
        nargs="+",
        default=list(),
        help="Keyword arguments to pass to the experiments. Note that kwargs are passed to all "
        "experiments specified.",
    )
    parser.add_argument(
        "--no-tests",
        required=False,
        action="store_true",
        help="Only test the main experiments, not those in tests/",
    )
    verbose_group = parser.add_mutually_exclusive_group()
    verbose_group.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Increase verbosity",
    )
    verbose_group.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="Decrease verbosity",
    )
    args = parser.parse_args()

    test_args = [
        "--module",
        "experiment_unittests",
    ]

    if args.experiments is not None:
        test_args.extend(
            [
                "--experiments",
                " ".join(args.experiments),
            ]
        )

    if args.kwargs is not None:
        test_args.extend(
            [
                "--kwargs",
                " ".join(args.kwargs),
            ]
        )

    if args.no_tests:
        test_args.append("--no-tests")
    if args.verbose:
        test_args.append("--verbose")
    if args.quiet:
        test_args.append("--quiet")

    os.environ["RADAR_ID"] = args.site_id
    from experiment_unittests import run_tests

    result = run_tests(test_args, buffer=True, print_results=True)

    if len(result.errors) + len(result.failures) != 0:
        exit(1)
