from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from dotenv import load_dotenv
from tinkoff.invest import Client


def main(cancel_all: bool = False):
    load_dotenv()

    token = os.getenv("TINVEST_TOKEN")
    account_id = os.getenv("TINVEST_SANDBOX_ACCOUNT_ID")

    if not token:
        raise RuntimeError("TINVEST_TOKEN is not set in .env")
    if not account_id:
        raise RuntimeError("TINVEST_SANDBOX_ACCOUNT_ID is not set in .env")

    with Client(token) as client:
        orders = client.sandbox.get_sandbox_orders(account_id=account_id).orders

        if not orders:
            print("No active sandbox orders.")
            return

        print("\nActive sandbox orders:")
        for order in orders:
            print(order)

        if cancel_all:
            print("\nCancelling all active sandbox orders...")
            for order in orders:
                try:
                    client.sandbox.cancel_sandbox_order(
                        account_id=account_id,
                        order_id=order.order_id,
                    )
                    print(f"Cancelled: {order.order_id}")
                except Exception as exc:
                    print(f"Failed to cancel {order.order_id}: {exc}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cancel-all",
        action="store_true",
        help="Cancel all active sandbox orders",
    )
    args = parser.parse_args()

    main(cancel_all=args.cancel_all)
