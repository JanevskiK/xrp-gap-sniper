def realized_pnl_total(con: sqlite3.Connection) -> float:
    """
    Realized PnL in quote asset from the trades table using average cost basis.
    BUY increases position + cost basis.
    SELL decreases position and realizes PnL against avg cost.
    """
    cur = con.execute(
        "SELECT ts, side, price, base_qty, quote_qty, fee_quote FROM trades ORDER BY ts ASC, id ASC"
    )

    pos_base = 0.0
    cost_quote = 0.0  # total cost basis of current position (in quote)

    realized = 0.0

    for ts, side, price, base_qty, quote_qty, fee_quote in cur.fetchall():
        fee_quote = float(fee_quote or 0.0)
        base_qty = float(base_qty)
        quote_qty = float(quote_qty)

        if side.upper() == "BUY":
            # You paid quote_qty (+ fee) and received base_qty
            pos_base += base_qty
            cost_quote += (quote_qty + fee_quote)

        elif side.upper() == "SELL":
            if pos_base <= 0:
                # selling without position; ignore or treat as error
                continue

            # Average cost per base unit
            avg_cost_per_base = cost_quote / pos_base

            # Cost basis removed for the sold qty
            removed_cost = avg_cost_per_base * base_qty

            # Proceeds net of fee
            proceeds = quote_qty - fee_quote

            realized += (proceeds - removed_cost)

            # Reduce position and remaining cost basis
            pos_base -= base_qty
            cost_quote -= removed_cost

            # Numerical guard
            if pos_base < 1e-12:
                pos_base = 0.0
                cost_quote = 0.0

    return realized


def realized_pnl_since(con: sqlite3.Connection, since_ts: int) -> float:
    """
    Realized PnL for SELL events with ts >= since_ts.
    Uses full history to build correct cost basis, then only counts realized PnL
    when the SELL happens after since_ts.
    """
    cur = con.execute(
        "SELECT ts, side, base_qty, quote_qty, fee_quote FROM trades ORDER BY ts ASC, id ASC"
    )

    pos_base = 0.0
    cost_quote = 0.0
    realized_since = 0.0

    for ts, side, base_qty, quote_qty, fee_quote in cur.fetchall():
        ts = int(ts)
        fee_quote = float(fee_quote or 0.0)
        base_qty = float(base_qty)
        quote_qty = float(quote_qty)

        if side.upper() == "BUY":
            pos_base += base_qty
            cost_quote += (quote_qty + fee_quote)

        elif side.upper() == "SELL":
            if pos_base <= 0:
                continue

            avg_cost_per_base = cost_quote / pos_base
            removed_cost = avg_cost_per_base * base_qty
            proceeds = quote_qty - fee_quote
            pnl = (proceeds - removed_cost)

            if ts >= since_ts:
                realized_since += pnl

            pos_base -= base_qty
            cost_quote -= removed_cost

            if pos_base < 1e-12:
                pos_base = 0.0
                cost_quote = 0.0

    return realized_since
