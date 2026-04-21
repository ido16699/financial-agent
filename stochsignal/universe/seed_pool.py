"""Static seed pool of ~300 US large/mid-cap tickers known to trade pre-2010.

Includes currently-listed names plus some notable delistings (e.g. LEH, BSC)
to reduce survivorship bias in the adaptive universe selection.

The adaptive selector (stochsignal.universe.adaptive) scores these by
point-in-time liquidity / volatility / data-completeness at each rebalance
date and picks the top N. A ticker only qualifies for a given as_of date
if it has price history up to that point — tickers that hadn't IPO'd or
that had been delisted are automatically filtered out by the scorer.
"""
from __future__ import annotations

# ~300 tickers — large/mid caps across major sectors, all trading pre-2010.
# Plus ~15 notable delistings / buyouts to reduce survivorship bias.
SEED_POOL: list[str] = [
    # Mega-cap tech
    "AAPL", "MSFT", "GOOGL", "GOOG", "AMZN", "META", "NVDA", "ORCL", "CSCO",
    "INTC", "IBM", "ADBE", "CRM", "TXN", "QCOM", "AMD", "AVGO", "MU",
    "ADI", "NXPI", "LRCX", "KLAC", "AMAT", "ADSK", "ANSS", "CDNS", "SNPS",
    "INTU", "CTSH", "ACN", "NOW", "WDAY", "EBAY", "NFLX", "BKNG", "EXPE",
    # Financials
    "JPM", "BAC", "WFC", "C", "GS", "MS", "BLK", "SCHW", "USB", "PNC",
    "AXP", "V", "MA", "COF", "TFC", "BK", "STT", "MET", "PRU", "AIG",
    "ALL", "TRV", "PGR", "HIG", "CB", "CME", "ICE", "SPGI", "MCO", "NDAQ",
    # Healthcare
    "JNJ", "UNH", "LLY", "PFE", "ABBV", "MRK", "ABT", "TMO", "DHR", "BMY",
    "AMGN", "GILD", "CVS", "MDT", "ISRG", "SYK", "BDX", "BSX", "EW", "REGN",
    "VRTX", "BIIB", "ZTS", "HUM", "CI", "HCA", "ANTM", "ELV", "MCK", "ABC",
    # Consumer discretionary
    "HD", "LOW", "MCD", "SBUX", "NKE", "TJX", "TGT", "DG", "DLTR", "ROST",
    "YUM", "CMG", "DPZ", "MAR", "HLT", "DIS", "F", "GM", "TSLA", "LVS",
    "WYNN", "MGM", "CCL", "RCL", "NCLH", "ULTA", "BBY", "ORLY", "AZO",
    # Consumer staples
    "PG", "KO", "PEP", "WMT", "COST", "MO", "PM", "MDLZ", "CL", "KMB",
    "GIS", "K", "HSY", "STZ", "KHC", "TAP", "SYY", "ADM", "KR", "WBA",
    # Industrials
    "BA", "CAT", "DE", "HON", "LMT", "RTX", "GD", "NOC", "GE", "MMM",
    "EMR", "ETN", "ITW", "PH", "ROK", "CMI", "PCAR", "UNP", "CSX", "NSC",
    "UPS", "FDX", "LUV", "DAL", "UAL", "AAL", "WM", "RSG", "EFX", "VRSK",
    # Energy
    "XOM", "CVX", "COP", "EOG", "SLB", "PXD", "OXY", "MPC", "PSX", "VLO",
    "HES", "DVN", "HAL", "BKR", "WMB", "KMI", "OKE", "ENB", "TRP", "EPD",
    # Materials
    "LIN", "APD", "SHW", "ECL", "DD", "DOW", "PPG", "NEM", "FCX", "NUE",
    "STLD", "VMC", "MLM", "BLL", "IP", "PKG", "CF", "MOS", "LYB",
    # Real estate
    "AMT", "PLD", "CCI", "EQIX", "PSA", "SPG", "O", "WELL", "DLR", "AVB",
    "EQR", "MAA", "ESS", "SBAC", "VTR", "HST", "REG", "FRT", "KIM",
    # Utilities
    "NEE", "DUK", "SO", "D", "AEP", "EXC", "SRE", "XEL", "WEC", "ED",
    "EIX", "PEG", "AWK", "PPL", "DTE", "AEE", "CMS", "CNP", "LNT", "ATO",
    # Telecom / Media
    "T", "VZ", "CMCSA", "TMUS", "CHTR", "DISH", "FOX", "FOXA", "PARA",
    "NWSA", "NWS", "OMC", "IPG", "WBD",
    # Additional mid-caps / specialty with pre-2010 history
    "COIN", "PYPL", "SQ", "SHOP",   # may fail pre-2010 filter; OK
    "CHKP", "TEVA", "NICE", "CEL",  # Israeli on US exchange (ADR)
    "BABA", "BIDU", "JD",           # Chinese ADRs
    "HPQ", "DELL", "STX", "WDC", "NTAP", "JNPR", "CTXS", "VMW",
    "ATVI", "EA", "TTWO", "MTCH",
    "ZM", "DOCU", "TEAM",           # may fail pre-2010 filter
    # Notable delistings / buyouts (for survivorship bias)
    "LEH",   # Lehman Brothers — bankrupt Sept 2008
    "BSC",   # Bear Stearns — collapsed March 2008
    "WM",    # Washington Mutual — bankrupt 2008 (reused ticker now Waste Mgmt — scorer will dedupe by date)
    "WB",    # Wachovia — acquired by Wells Fargo 2008
    "CIT",   # CIT Group — bankrupt 2009
    "AIG",   # AIG — bailout 2008, heavily diluted
    "COP",   # ConocoPhillips (still exists, but many energy firms acquired)
    "SUN",   # Sunoco — bought by ETP
    "XTO",   # XTO Energy — bought by Exxon 2010
    "NOVL",  # Novell — acquired 2011
    "PALM",  # Palm Inc — acquired by HP 2010
    "SUNE",  # SunEdison — bankrupt 2016
    "RSH",   # RadioShack — bankrupt 2015
    "JCP",   # JCPenney — bankrupt 2020
    "SHLD",  # Sears Holdings — bankrupt 2018
    "DDS",   # Dillard's (still trading, OK)
    "EK",    # Eastman Kodak — bankrupt 2012
    "GM.OLD",  # Old GM (pre-2009 bankruptcy) — yfinance may not have
    "BAC",   # BofA near collapse 2008
    "GE",    # General Electric — heavily restructured
    "F",     # Ford — bailout era
]

# De-duplicate while preserving order
def _dedupe(items: list[str]) -> list[str]:
    seen = set()
    out = []
    for x in items:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out

SEED_POOL = _dedupe(SEED_POOL)
