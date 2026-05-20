from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from dotenv import load_dotenv
from tinkoff.invest import Client, MoneyValue


def main():
    load_dotenv()

    token = os.getenv("TINVEST_TOKEN")
    if not token:
        raise RuntimeError("TINVEST_TOKEN is not set in .env")

    with Client(token) as client:
        accounts = client.sandbox.get_sandbox_accounts().accounts

        if accounts:
            account_id = accounts[0].id
            print(f"Existing sandbox account: {account_id}")
        else:
            response = client.sandbox.open_sandbox_account()
            account_id = response.account_id
            print(f"Created sandbox account: {account_id}")

        # Пополняем песочницу виртуальными рублями.
        client.sandbox.sandbox_pay_in(
            account_id=account_id,
            amount=MoneyValue(
                currency="rub",
                units=100_000,
                nano=0,
            ),
        )

        portfolio = client.sandbox.get_sandbox_portfolio(account_id=account_id)

        print("\nSandbox portfolio:")
        print(portfolio)

        print("\nAdd this to .env:")
        print(f"TINVEST_SANDBOX_ACCOUNT_ID={account_id}")


if __name__ == "__main__":
    main()
