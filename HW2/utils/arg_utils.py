import argparse

from schemas.args import Args


def parse_arguments() -> Args:
    parser = argparse.ArgumentParser()
    parser.add_argument('--window_size', type=int, default=10, choices=[10, 30, 60])
    parser.add_argument('--cms_width', type=int, default=2048, choices=[2048, 8192])
    parser.add_argument('--cs_rows', type=int, default=5, choices=[5, 7, 9])
    parser.add_argument('--cs_buckets', type=int, default=2048, choices=[2048, 8192])
    parser.add_argument('--ams_rows', type=int, default=16, choices=[16, 64, 256])
    
    args = parser.parse_args()
    return Args(
        window_size=args.window_size,
        cms_width=args.cms_width,
        cs_rows=args.cs_rows,
        cs_buckets=args.cs_buckets,
        ams_rows=args.ams_rows
    )
