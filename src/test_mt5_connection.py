"""Read-only diagnostic for the already logged-in MetaTrader 5 terminal."""

from __future__ import annotations

from datetime import datetime, timezone

import MetaTrader5 as mt5


SYMBOL = "EURUSD"
PIP_SIZE = 0.0001


def utc_from_milliseconds(time_msc: int) -> datetime:
    return datetime.fromtimestamp(time_msc / 1_000, tz=timezone.utc)


def main() -> None:
    initialized = False
    try:
        initialized = bool(mt5.initialize())
        if not initialized:
            raise RuntimeError(f"mt5.initialize() failed: {mt5.last_error()}")

        terminal = mt5.terminal_info()
        account = mt5.account_info()
        if terminal is None:
            raise RuntimeError(f"mt5.terminal_info() failed: {mt5.last_error()}")
        if account is None:
            raise RuntimeError(f"mt5.account_info() failed: {mt5.last_error()}")
        if not mt5.symbol_select(SYMBOL, True):
            raise RuntimeError(f"mt5.symbol_select({SYMBOL!r}, True) failed: {mt5.last_error()}")

        symbol = mt5.symbol_info(SYMBOL)
        tick = mt5.symbol_info_tick(SYMBOL)
        if symbol is None:
            raise RuntimeError(f"mt5.symbol_info({SYMBOL!r}) failed: {mt5.last_error()}")
        if tick is None:
            raise RuntimeError(f"mt5.symbol_info_tick({SYMBOL!r}) failed: {mt5.last_error()}")

        bid = float(tick.bid)
        ask = float(tick.ask)
        mid = (bid + ask) / 2
        spread = ask - bid
        tick_time = utc_from_milliseconds(int(tick.time_msc))
        tick_age = datetime.now(timezone.utc) - tick_time

        print(f"MT5 package version: {mt5.__version__}")
        print(f"Terminal connected: {bool(terminal.connected)}")
        print(f"Account server: {account.server}")
        print(f"Symbol: {symbol.name}")
        print(f"UTC tick time: {tick_time.isoformat(timespec='milliseconds')}")
        print(f"Bid: {bid:.{symbol.digits}f}")
        print(f"Ask: {ask:.{symbol.digits}f}")
        print(f"Mid: {mid:.{symbol.digits}f}")
        print(f"Spread: {spread:.{symbol.digits}f}")
        print(f"Spread in pips: {spread / PIP_SIZE:.3f}")
        print(f"Tick age seconds: {tick_age.total_seconds():.3f}")
    except RuntimeError as exc:
        raise SystemExit(f"ERROR: {exc}") from exc
    finally:
        # Safe even after a failed initialize, and guarantees no lingering API session.
        mt5.shutdown()


if __name__ == "__main__":
    main()
