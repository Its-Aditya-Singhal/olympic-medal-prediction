"""Map Olympic NOC codes (and Wikipedia country names) to World Bank ISO3 codes.

Most NOC codes equal the ISO3 code, so only the differences are listed.
A few NOC codes collide with a *different* ISO3 country (BRN = Bahrain but
ISO BRN = Brunei; CHI = Chile but ISO CHI = Channel Islands), so those are
listed too. Historical teams are folded into their successor states so that
"medals last Games" carries over (e.g. URS 1988 -> EUN 1992 -> RUS 1996).
"""

NOC_TO_ISO3 = {
    # historical / successor states
    "URS": "RUS", "EUN": "RUS", "ROC": "RUS",
    "FRG": "DEU", "GDR": "DEU", "GER": "DEU",
    "TCH": "CZE",
    "YUG": "SRB", "SCG": "SRB",
    "YAR": "YEM", "YMD": "YEM",
    # NOC code differs from ISO3
    "ALG": "DZA", "ANG": "AGO", "ANT": "ATG", "ARU": "ABW", "ASA": "ASM",
    "BAH": "BHS", "BAN": "BGD", "BAR": "BRB", "BER": "BMU", "BHU": "BTN",
    "BIZ": "BLZ", "BOT": "BWA", "BRN": "BHR", "BRU": "BRN", "BUL": "BGR",
    "BUR": "BFA", "CAM": "KHM", "CAY": "CYM", "CGO": "COG", "CHA": "TCD",
    "CHI": "CHL", "CRC": "CRI", "CRO": "HRV", "DEN": "DNK", "ESA": "SLV",
    "FIJ": "FJI", "GAM": "GMB", "GBS": "GNB", "GEQ": "GNQ", "GRE": "GRC",
    "GRN": "GRD", "GUA": "GTM", "GUI": "GIN", "HAI": "HTI", "HON": "HND",
    "INA": "IDN", "IRI": "IRN", "ISV": "VIR", "IVB": "VGB", "KOS": "XKX",
    "KSA": "SAU", "KUW": "KWT", "LAT": "LVA", "LBA": "LBY", "LES": "LSO",
    "LIB": "LBN", "MAD": "MDG", "MAS": "MYS", "MAW": "MWI", "MGL": "MNG",
    "MON": "MCO", "MRI": "MUS", "MTN": "MRT", "MYA": "MMR", "NCA": "NIC",
    "NED": "NLD", "NEP": "NPL", "NGR": "NGA", "NIG": "NER", "OMA": "OMN",
    "PAR": "PRY", "PHI": "PHL", "PLE": "PSE", "POR": "PRT", "PUR": "PRI",
    "RSA": "ZAF", "SAM": "WSM", "SEY": "SYC", "SIN": "SGP", "SKN": "KNA",
    "SLO": "SVN", "SOL": "SLB", "SRI": "LKA", "SUD": "SDN", "SUI": "CHE",
    "TAN": "TZA", "TGA": "TON", "TOG": "TGO", "TPE": "TWN", "UAE": "ARE",
    "URU": "URY", "VAN": "VUT", "VIE": "VNM", "VIN": "VCT", "ZAM": "ZMB",
    "ZIM": "ZWE",
}

# Teams with no country to attach economic data to.
DROP_NOCS = {"IOA", "ROT", "EOR", "AHO", "UNK"}


def noc_to_iso3(noc: str):
    if noc in DROP_NOCS:
        return None
    return NOC_TO_ISO3.get(noc, noc)


# Wikipedia country names (Tokyo 2020 pages) whose World Bank name differs.
NAME_TO_ISO3 = {
    "United States": "USA", "Great Britain": "GBR", "ROC": "RUS",
    "South Korea": "KOR", "North Korea": "PRK", "Iran": "IRN",
    "Chinese Taipei": "TWN", "Hong Kong": "HKG", "Czech Republic": "CZE",
    "Slovakia": "SVK", "Egypt": "EGY", "Venezuela": "VEN", "Turkey": "TUR",
    "Russia": "RUS", "Syria": "SYR", "Kyrgyzstan": "KGZ", "Bahamas": "BHS",
    "Gambia": "GMB", "The Gambia": "GMB", "Ivory Coast": "CIV",
    "Côte d'Ivoire": "CIV", "Cape Verde": "CPV", "Congo": "COG",
    "Republic of the Congo": "COG", "DR Congo": "COD",
    "Democratic Republic of the Congo": "COD", "Laos": "LAO",
    "Micronesia": "FSM", "Federated States of Micronesia": "FSM",
    "Saint Kitts and Nevis": "KNA", "Saint Lucia": "LCA",
    "Saint Vincent and the Grenadines": "VCT", "Vietnam": "VNM",
    "Yemen": "YEM", "Brunei": "BRN", "Macedonia": "MKD",
    "North Macedonia": "MKD", "Moldova": "MDA", "Palestine": "PSE",
    "Kosovo": "XKX", "Eswatini": "SWZ", "Swaziland": "SWZ",
    "Virgin Islands": "VIR", "United States Virgin Islands": "VIR",
    "British Virgin Islands": "VGB", "Sao Tome and Principe": "STP",
    "São Tomé and Príncipe": "STP", "Timor-Leste": "TLS", "East Timor": "TLS",
    "Somalia": "SOM", "Tanzania": "TZA", "Bolivia": "BOL",
    "Cook Islands": "COK", "Hong Kong, China": "HKG",
    "Puerto Rico": "PRI", "Nauru": "NRU",
}
