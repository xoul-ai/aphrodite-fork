import argparse

import aphrodite.endpoints.cli.benchmark.latency
import aphrodite.endpoints.cli.benchmark.serve
import aphrodite.endpoints.cli.benchmark.throughput
from aphrodite.common.utils import FlexibleArgumentParser
from aphrodite.endpoints.cli.types import CLISubcommand

BENCHMARK_CMD_MODULES = [
    aphrodite.endpoints.cli.benchmark.latency,
    aphrodite.endpoints.cli.benchmark.serve,
    aphrodite.endpoints.cli.benchmark.throughput,
]


class BenchmarkSubcommand(CLISubcommand):
    """ The `bench` subcommand for the Aphrodite CLI. """

    def __init__(self):
        self.name = "bench"
        super().__init__()

    @staticmethod
    def cmd(args: argparse.Namespace) -> None:
        args.dispatch_function(args)

    def validate(self, args: argparse.Namespace) -> None:
        if args.bench_type in self.cmds:
            self.cmds[args.bench_type].validate(args)

    def subparser_init(
            self,
            subparsers: argparse._SubParsersAction) -> FlexibleArgumentParser:
        bench_parser = subparsers.add_parser(
            "bench",
            help="Aphrodite bench subcommand.",
            description="Aphrodite bench subcommand.",
            usage="aphrodite bench <bench_type> [options]")
        bench_subparsers = bench_parser.add_subparsers(required=True,
                                                       dest="bench_type")
        self.cmds = {}
        for cmd_module in BENCHMARK_CMD_MODULES:
            new_cmds = cmd_module.cmd_init()
            for cmd in new_cmds:
                cmd.subparser_init(bench_subparsers).set_defaults(
                    dispatch_function=cmd.cmd)
                self.cmds[cmd.name] = cmd
        return bench_parser


def cmd_init() -> list[CLISubcommand]:
    return [BenchmarkSubcommand()]
