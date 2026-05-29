"""V5 v1 paper-trading runtime.

Modules:
- broker         : BrokerClient abstraction + MockBroker + UpstoxBroker stub
- online_features: chain / futures feature compute at minute t from broker quotes
- ledger         : paper trade ledger schema + IO
- decision       : per-minute decision pipeline (used by scripts/v5_minute_decision.py)
- data_quality   : minute-session completeness checks for futures paper/live ops
- risk           : live-readiness risk governor for order tickets
- reconcile      : paper-ledger to order-ticket reconciliation
- ops_report     : JSON reports for daily paper-trading operations
"""
