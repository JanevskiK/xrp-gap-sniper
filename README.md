# XRP Gap Sniper 🤖

A Python trading bot that captures bid/ask spread gaps on XRP/USD
using post-only maker orders on a single exchange.

## Strategy
- Scan order book for wide spreads
- Place maker BUY inside bid
- On fill, place maker SELL inside ask
- Cancel fast if not filled (5–10s)
- No long-term holding
