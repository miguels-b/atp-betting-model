"""
=============================================================================
 MÓDULO DE SCRAPING DE CUOTAS — Playwright + BeautifulSoup
 
 Fuentes soportadas:
   1. OddsPortal (principal) — cuotas de ~20 casas, estructura más estable
   2. Oddschecker (fallback) — comparador UK, agresivo contra bots
 
 Requisitos:
   pip install playwright beautifulsoup4 lxml
   playwright install chromium
 
 El módulo expone fetch_matches() como interfaz única.
=============================================================================
"""
import re
import time
import random
import numpy as np
from datetime import datetime
from bs4 import BeautifulSoup
from collections import defaultdict

# =====================================================================
# CONFIGURACIÓN DEL SCRAPER
# =====================================================================
HEADLESS = True           # False para debug visual
SLOW_MO = 0              # ms entre acciones (50-100 para debug)
REQUEST_DELAY = (1.5, 3.0)  # rango de delay aleatorio entre páginas (segundos)
PAGE_TIMEOUT = 30000      # timeout de carga de página (ms)

# =====================================================================
# Mapeo de torneos → URLs y metadata
# =====================================================================
TOURNAMENT_CONFIG = {
    # ══════════════════════════════════════════════════════════════════
    # GRAND SLAMS
    # ══════════════════════════════════════════════════════════════════
    "australian-open": {
        "surface": "Hard", "court": "Outdoor", "series": "Grand Slam",
        "name": "Australian Open",
        "oddsportal": "https://www.oddsportal.com/tennis/australia/atp-australian-open/",
    },
    "french-open": {
        "surface": "Clay", "court": "Outdoor", "series": "Grand Slam",
        "name": "Roland Garros",
        "oddsportal": "https://www.oddsportal.com/tennis/france/atp-french-open/",
    },
    "wimbledon": {
        "surface": "Grass", "court": "Outdoor", "series": "Grand Slam",
        "name": "Wimbledon",
        "oddsportal": "https://www.oddsportal.com/tennis/united-kingdom/atp-wimbledon/",
    },
    "us-open": {
        "surface": "Hard", "court": "Outdoor", "series": "Grand Slam",
        "name": "US Open",
        "oddsportal": "https://www.oddsportal.com/tennis/usa/atp-us-open/",
    },
    # ══════════════════════════════════════════════════════════════════
    # MASTERS 1000
    # ══════════════════════════════════════════════════════════════════
    "atp-miami": {
        "surface": "Hard", "court": "Outdoor", "series": "Masters 1000",
        "name": "Miami Open",
        "oddsportal": "https://www.oddsportal.com/tennis/usa/atp-miami/",
    },
    "atp-indian-wells": {
        "surface": "Hard", "court": "Outdoor", "series": "Masters 1000",
        "name": "Indian Wells",
        "oddsportal": "https://www.oddsportal.com/tennis/usa/atp-indian-wells/",
    },
    "atp-monte-carlo": {
        "surface": "Clay", "court": "Outdoor", "series": "Masters 1000",
        "name": "Monte Carlo",
        "oddsportal": "https://www.oddsportal.com/tennis/monaco/atp-monte-carlo/",
    },
    "atp-madrid": {
        "surface": "Clay", "court": "Outdoor", "series": "Masters 1000",
        "name": "Madrid Open",
        "oddsportal": "https://www.oddsportal.com/tennis/spain/atp-madrid/",
    },
    "atp-rome": {
        "surface": "Clay", "court": "Outdoor", "series": "Masters 1000",
        "name": "Rome Masters",
        "oddsportal": "https://www.oddsportal.com/tennis/italy/atp-rome/",
    },
    "atp-shanghai": {
        "surface": "Hard", "court": "Outdoor", "series": "Masters 1000",
        "name": "Shanghai Masters",
        "oddsportal": "https://www.oddsportal.com/tennis/china/atp-shanghai/",
    },
    "atp-paris": {
        "surface": "Hard", "court": "Indoor", "series": "Masters 1000",
        "name": "Paris Masters",
        "oddsportal": "https://www.oddsportal.com/tennis/france/atp-paris/",
    },
    "atp-canada": {
        "surface": "Hard", "court": "Outdoor", "series": "Masters 1000",
        "name": "Canadian Open",
        "oddsportal": "https://www.oddsportal.com/tennis/canada/atp-montreal/",
    },
    "atp-cincinnati": {
        "surface": "Hard", "court": "Outdoor", "series": "Masters 1000",
        "name": "Cincinnati Open",
        "oddsportal": "https://www.oddsportal.com/tennis/usa/atp-cincinnati/",
    },
    # ══════════════════════════════════════════════════════════════════
    # ATP 500
    # ══════════════════════════════════════════════════════════════════
    "atp-barcelona": {
        "surface": "Clay", "court": "Outdoor", "series": "ATP500",
        "name": "Barcelona",
        "oddsportal": "https://www.oddsportal.com/tennis/spain/atp-barcelona/",
    },
    "atp-dubai": {
        "surface": "Hard", "court": "Outdoor", "series": "ATP500",
        "name": "Dubai",
        "oddsportal": "https://www.oddsportal.com/tennis/united-arab-emirates/atp-dubai/",
    },
    "atp-rotterdam": {
        "surface": "Hard", "court": "Indoor", "series": "ATP500",
        "name": "Rotterdam",
        "oddsportal": "https://www.oddsportal.com/tennis/netherlands/atp-rotterdam/",
    },
    "atp-halle": {
        "surface": "Grass", "court": "Outdoor", "series": "ATP500",
        "name": "Halle",
        "oddsportal": "https://www.oddsportal.com/tennis/germany/atp-halle/",
    },
    "queens-club": {
        "surface": "Grass", "court": "Outdoor", "series": "ATP500",
        "name": "Queen's Club",
        "oddsportal": "https://www.oddsportal.com/tennis/united-kingdom/atp-london/",
    },
    "atp-hamburg": {
        "surface": "Clay", "court": "Outdoor", "series": "ATP500",
        "name": "Hamburg",
        "oddsportal": "https://www.oddsportal.com/tennis/germany/atp-hamburg/",
    },
    "atp-vienna": {
        "surface": "Hard", "court": "Indoor", "series": "ATP500",
        "name": "Vienna",
        "oddsportal": "https://www.oddsportal.com/tennis/austria/atp-vienna/",
    },
    "atp-basel": {
        "surface": "Hard", "court": "Indoor", "series": "ATP500",
        "name": "Basel",
        "oddsportal": "https://www.oddsportal.com/tennis/switzerland/atp-basel/",
    },
    "atp-acapulco": {
        "surface": "Hard", "court": "Outdoor", "series": "ATP500",
        "name": "Acapulco",
        "oddsportal": "https://www.oddsportal.com/tennis/mexico/atp-acapulco/",
    },
    "atp-washington": {
        "surface": "Hard", "court": "Outdoor", "series": "ATP500",
        "name": "Washington",
        "oddsportal": "https://www.oddsportal.com/tennis/usa/atp-washington/",
    },
    "atp-beijing": {
        "surface": "Hard", "court": "Outdoor", "series": "ATP500",
        "name": "Beijing",
        "oddsportal": "https://www.oddsportal.com/tennis/china/atp-beijing/",
    },
    "atp-rio": {
        "surface": "Clay", "court": "Outdoor", "series": "ATP500",
        "name": "Rio Open",
        "oddsportal": "https://www.oddsportal.com/tennis/brazil/atp-rio-de-janeiro/",
    },
    "atp-tokyo": {
        "surface": "Hard", "court": "Outdoor", "series": "ATP500",
        "name": "Tokyo (Rakuten)",
        "oddsportal": "https://www.oddsportal.com/tennis/japan/atp-tokyo/",
    },
    # ══════════════════════════════════════════════════════════════════
    # ATP 250
    # ══════════════════════════════════════════════════════════════════
    "atp-brisbane": {
        "surface": "Hard", "court": "Outdoor", "series": "ATP250",
        "name": "Brisbane",
        "oddsportal": "https://www.oddsportal.com/tennis/australia/atp-brisbane/",
    },
    "atp-adelaide": {
        "surface": "Hard", "court": "Outdoor", "series": "ATP250",
        "name": "Adelaide",
        "oddsportal": "https://www.oddsportal.com/tennis/australia/atp-adelaide/",
    },
    "atp-auckland": {
        "surface": "Hard", "court": "Outdoor", "series": "ATP250",
        "name": "Auckland",
        "oddsportal": "https://www.oddsportal.com/tennis/new-zealand/atp-auckland/",
    },
    "atp-montpellier": {
        "surface": "Hard", "court": "Indoor", "series": "ATP250",
        "name": "Montpellier",
        "oddsportal": "https://www.oddsportal.com/tennis/france/atp-montpellier/",
    },
    "atp-marseille": {
        "surface": "Hard", "court": "Indoor", "series": "ATP250",
        "name": "Marseille",
        "oddsportal": "https://www.oddsportal.com/tennis/france/atp-marseille/",
    },
    "atp-buenos-aires": {
        "surface": "Clay", "court": "Outdoor", "series": "ATP250",
        "name": "Buenos Aires",
        "oddsportal": "https://www.oddsportal.com/tennis/argentina/atp-buenos-aires/",
    },
    "atp-santiago": {
        "surface": "Clay", "court": "Outdoor", "series": "ATP250",
        "name": "Santiago",
        "oddsportal": "https://www.oddsportal.com/tennis/chile/atp-santiago/",
    },
    "atp-dallas": {
        "surface": "Hard", "court": "Indoor", "series": "ATP250",
        "name": "Dallas",
        "oddsportal": "https://www.oddsportal.com/tennis/usa/atp-dallas/",
    },
    "atp-doha": {
        "surface": "Hard", "court": "Outdoor", "series": "ATP250",
        "name": "Doha",
        "oddsportal": "https://www.oddsportal.com/tennis/qatar/atp-doha/",
    },
    "atp-houston": {
        "surface": "Clay", "court": "Outdoor", "series": "ATP250",
        "name": "Houston",
        "oddsportal": "https://www.oddsportal.com/tennis/usa/atp-houston/",
    },
    "atp-marrakech": {
        "surface": "Clay", "court": "Outdoor", "series": "ATP250",
        "name": "Marrakech",
        "oddsportal": "https://www.oddsportal.com/tennis/morocco/atp-marrakech/",
    },
    "atp-bucharest": {
        "surface": "Clay", "court": "Outdoor", "series": "ATP250",
        "name": "Bucharest",
        "oddsportal": "https://www.oddsportal.com/tennis/romania/atp-bucharest/",
    },
    "atp-lyon": {
        "surface": "Clay", "court": "Outdoor", "series": "ATP250",
        "name": "Lyon",
        "oddsportal": "https://www.oddsportal.com/tennis/france/atp-lyon/",
    },
    "atp-geneva": {
        "surface": "Clay", "court": "Outdoor", "series": "ATP250",
        "name": "Geneva",
        "oddsportal": "https://www.oddsportal.com/tennis/switzerland/atp-geneva/",
    },
    "atp-stuttgart": {
        "surface": "Grass", "court": "Outdoor", "series": "ATP250",
        "name": "Stuttgart",
        "oddsportal": "https://www.oddsportal.com/tennis/germany/atp-stuttgart/",
    },
    "atp-hertogenbosch": {
        "surface": "Grass", "court": "Outdoor", "series": "ATP250",
        "name": "'s-Hertogenbosch",
        "oddsportal": "https://www.oddsportal.com/tennis/netherlands/atp-s-hertogenbosch/",
    },
    "atp-mallorca": {
        "surface": "Grass", "court": "Outdoor", "series": "ATP250",
        "name": "Mallorca",
        "oddsportal": "https://www.oddsportal.com/tennis/spain/atp-mallorca/",
    },
    "atp-eastbourne": {
        "surface": "Grass", "court": "Outdoor", "series": "ATP250",
        "name": "Eastbourne",
        "oddsportal": "https://www.oddsportal.com/tennis/united-kingdom/atp-eastbourne/",
    },
    "atp-bastad": {
        "surface": "Clay", "court": "Outdoor", "series": "ATP250",
        "name": "Bastad",
        "oddsportal": "https://www.oddsportal.com/tennis/sweden/atp-bastad/",
    },
    "atp-umag": {
        "surface": "Clay", "court": "Outdoor", "series": "ATP250",
        "name": "Umag",
        "oddsportal": "https://www.oddsportal.com/tennis/croatia/atp-umag/",
    },
    "atp-gstaad": {
        "surface": "Clay", "court": "Outdoor", "series": "ATP250",
        "name": "Gstaad",
        "oddsportal": "https://www.oddsportal.com/tennis/switzerland/atp-gstaad/",
    },
    "atp-kitzbuhel": {
        "surface": "Clay", "court": "Outdoor", "series": "ATP250",
        "name": "Kitzbühel",
        "oddsportal": "https://www.oddsportal.com/tennis/austria/atp-kitzbuhel/",
    },
    "atp-atlanta": {
        "surface": "Hard", "court": "Outdoor", "series": "ATP250",
        "name": "Atlanta",
        "oddsportal": "https://www.oddsportal.com/tennis/usa/atp-atlanta/",
    },
    "atp-winston-salem": {
        "surface": "Hard", "court": "Outdoor", "series": "ATP250",
        "name": "Winston-Salem",
        "oddsportal": "https://www.oddsportal.com/tennis/usa/atp-winston-salem/",
    },
    "atp-chengdu": {
        "surface": "Hard", "court": "Outdoor", "series": "ATP250",
        "name": "Chengdu",
        "oddsportal": "https://www.oddsportal.com/tennis/china/atp-chengdu/",
    },
    "atp-zhuhai": {
        "surface": "Hard", "court": "Outdoor", "series": "ATP250",
        "name": "Zhuhai",
        "oddsportal": "https://www.oddsportal.com/tennis/china/atp-zhuhai/",
    },
    "atp-stockholm": {
        "surface": "Hard", "court": "Indoor", "series": "ATP250",
        "name": "Stockholm",
        "oddsportal": "https://www.oddsportal.com/tennis/sweden/atp-stockholm/",
    },
    "atp-antwerp": {
        "surface": "Hard", "court": "Indoor", "series": "ATP250",
        "name": "Antwerp",
        "oddsportal": "https://www.oddsportal.com/tennis/belgium/atp-antwerp/",
    },
    "atp-metz": {
        "surface": "Hard", "court": "Indoor", "series": "ATP250",
        "name": "Metz",
        "oddsportal": "https://www.oddsportal.com/tennis/france/atp-metz/",
    },
    "atp-sofia": {
        "surface": "Hard", "court": "Indoor", "series": "ATP250",
        "name": "Sofia",
        "oddsportal": "https://www.oddsportal.com/tennis/bulgaria/atp-sofia/",
    },
    "atp-belgrade": {
        "surface": "Hard", "court": "Indoor", "series": "ATP250",
        "name": "Belgrade",
        "oddsportal": "https://www.oddsportal.com/tennis/serbia/atp-belgrade/",
    },
    # ══════════════════════════════════════════════════════════════════
    # ATP FINALS / MASTERS CUP
    # ══════════════════════════════════════════════════════════════════
    "atp-finals": {
        "surface": "Hard", "court": "Indoor", "series": "Masters Cup",
        "name": "ATP Finals",
        "oddsportal": "https://www.oddsportal.com/tennis/world/atp-finals-turin/",
    },
}

# =====================================================================
# Mapeo de nombres → formato tennis-data.co.uk
# =====================================================================
PLAYER_NAME_MAP = {
    # Nombres completos (OddsPortal usa "Sinner J." o "Jannik Sinner")
    "Jannik Sinner": "Sinner J.", "Sinner J.": "Sinner J.",
    "Carlos Alcaraz": "Alcaraz C.", "Alcaraz C.": "Alcaraz C.",
    "Alexander Zverev": "Zverev A.", "Zverev A.": "Zverev A.",
    "Novak Djokovic": "Djokovic N.", "Djokovic N.": "Djokovic N.",
    "Daniil Medvedev": "Medvedev D.", "Medvedev D.": "Medvedev D.",
    "Taylor Fritz": "Fritz T.", "Fritz T.": "Fritz T.",
    "Tommy Paul": "Paul T.", "Paul T.": "Paul T.",
    "Ben Shelton": "Shelton B.", "Shelton B.": "Shelton B.",
    "Felix Auger-Aliassime": "Auger-Aliassime F.", "Auger-Aliassime F.": "Auger-Aliassime F.",
    "Stefanos Tsitsipas": "Tsitsipas S.", "Tsitsipas S.": "Tsitsipas S.",
    "Andrey Rublev": "Rublev A.", "Rublev A.": "Rublev A.",
    "Casper Ruud": "Ruud C.", "Ruud C.": "Ruud C.",
    "Hubert Hurkacz": "Hurkacz H.", "Hurkacz H.": "Hurkacz H.",
    "Alex De Minaur": "De Minaur A.", "Alex de Minaur": "De Minaur A.",
    "de Minaur A.": "De Minaur A.", "De Minaur A.": "De Minaur A.",
    "Frances Tiafoe": "Tiafoe F.", "Tiafoe F.": "Tiafoe F.",
    "Lorenzo Musetti": "Musetti L.", "Musetti L.": "Musetti L.",
    "Francisco Cerundolo": "Cerundolo F.", "Cerundolo F.": "Cerundolo F.",
    "Holger Rune": "Rune H.", "Rune H.": "Rune H.",
    "Jack Draper": "Draper J.", "Draper J.": "Draper J.",
    "Ugo Humbert": "Humbert U.", "Humbert U.": "Humbert U.",
    "Sebastian Korda": "Korda S.", "Korda S.": "Korda S.",
    "Arthur Fils": "Fils A.", "Fils A.": "Fils A.",
    "Rafael Nadal": "Nadal R.", "Nadal R.": "Nadal R.",
    "Grigor Dimitrov": "Dimitrov G.", "Dimitrov G.": "Dimitrov G.",
    "Cameron Norrie": "Norrie C.", "Norrie C.": "Norrie C.",
    "Alexander Bublik": "Bublik A.", "Bublik A.": "Bublik A.",
    "Flavio Cobolli": "Cobolli F.", "Cobolli F.": "Cobolli F.",
    "Jakub Mensik": "Mensik J.", "Mensik J.": "Mensik J.",
    "Alejandro Tabilo": "Tabilo A.", "Tabilo A.": "Tabilo A.",
    "Karen Khachanov": "Khachanov K.", "Khachanov K.": "Khachanov K.",
    "Jiri Lehecka": "Lehecka J.", "Lehecka J.": "Lehecka J.",
    "Matteo Berrettini": "Berrettini M.", "Berrettini M.": "Berrettini M.",
    "Alex Michelsen": "Michelsen A.", "Michelsen A.": "Michelsen A.",
    "Learner Tien": "Tien L.", "Tien L.": "Tien L.",
    "Brandon Nakashima": "Nakashima B.", "Nakashima B.": "Nakashima B.",
    "Tallon Griekspoor": "Griekspoor T.", "Griekspoor T.": "Griekspoor T.",
    "Denis Shapovalov": "Shapovalov D.", "Shapovalov D.": "Shapovalov D.",
    "Gael Monfils": "Monfils G.", "Monfils G.": "Monfils G.",
    "Stan Wawrinka": "Wawrinka S.", "Wawrinka S.": "Wawrinka S.",
    "Nicolas Jarry": "Jarry N.", "Jarry N.": "Jarry N.",
    "Tomas Machac": "Machac T.", "Machac T.": "Machac T.",
    "Jordan Thompson": "Thompson J.", "Thompson J.": "Thompson J.",
    "Nuno Borges": "Borges N.", "Borges N.": "Borges N.",
    "Jan-Lennard Struff": "Struff J.L.", "Struff J.": "Struff J.L.",
    "Roberto Bautista Agut": "Bautista Agut R.", "Bautista Agut R.": "Bautista Agut R.",
    "Tomas Martin Etcheverry": "Etcheverry T.", "Etcheverry T.": "Etcheverry T.",
    "Etcheverry T. M.": "Etcheverry T.",
    "Arthur Rinderknech": "Rinderknech A.", "Rinderknech A.": "Rinderknech A.",
    "Martin Landaluce": "Landaluce M.", "Landaluce M.": "Landaluce M.",
    "Terence Atmane": "Atmane T.", "Atmane T.": "Atmane T.",
    "Corentin Moutet": "Moutet C.", "Moutet C.": "Moutet C.",
}


def normalize_player_name(raw_name):
    """
    Normaliza un nombre de jugador al formato tennis-data.co.uk ("Apellido I.").
    Soporta múltiples formatos de entrada:
      - "Sinner J." → "Sinner J."           (OddsPortal format, ya correcto)
      - "Etcheverry T. M." → "Etcheverry T." (OddsPortal con segundo nombre)
      - "Jannik Sinner" → "Sinner J."        (nombre completo)
      - "J. Sinner" → "Sinner J."
      - "jannik-sinner" (slug) → "Sinner J."
    """
    name = raw_name.strip()
    
    # 1. Lookup directo
    if name in PLAYER_NAME_MAP:
        return PLAYER_NAME_MAP[name]
    
    # 2. Es un slug? (contiene guiones, sin espacios)
    if "-" in name and " " not in name:
        return _slug_to_player(name)
    
    # 3. Formato OddsPortal: "Apellido I." o "Apellido I. M." (con inicial extra)
    #    Ejemplos: "Sinner J.", "Etcheverry T. M.", "De Minaur A."
    m = re.match(r'^(.+?)\s+([A-Z])\.\s*([A-Z]\.)?$', name)
    if m:
        surname = m.group(1)
        initial = m.group(2)
        result = f"{surname} {initial}."
        return PLAYER_NAME_MAP.get(result, result)
    
    # 4. Formato "I. Apellido" → "Apellido I."
    m = re.match(r'^([A-Z])\.\s+(.+)$', name)
    if m:
        result = f"{m.group(2)} {m.group(1)}."
        return PLAYER_NAME_MAP.get(result, result)
    
    # 5. Formato "Nombre Apellido" → "Apellido N."
    parts = name.split()
    if len(parts) >= 2:
        # Manejar apellidos compuestos con guión
        if any("-" in p for p in parts):
            hyphenated = [p for p in parts if "-" in p]
            others = [p for p in parts if "-" not in p]
            if hyphenated and others:
                result = f"{hyphenated[0]} {others[0][0]}."
                return PLAYER_NAME_MAP.get(result, result)
        
        # Caso especial: 3+ palabras (e.g. "Roberto Bautista Agut")
        if len(parts) >= 3:
            candidate = f"{' '.join(parts[1:])} {parts[0][0]}."
            if candidate in PLAYER_NAME_MAP:
                return PLAYER_NAME_MAP[candidate]
        
        # Caso normal
        result = f"{parts[-1]} {parts[0][0]}."
        return PLAYER_NAME_MAP.get(result, result)
    
    return name


def _slug_to_player(slug):
    """Convierte slug de URL a nombre de jugador."""
    if slug in PLAYER_NAME_MAP:
        return PLAYER_NAME_MAP[slug]
    
    parts = slug.split("-")
    if len(parts) >= 2:
        if "auger" in parts and "aliassime" in parts:
            return "Auger-Aliassime F."
        if len(parts) == 3:
            return f"{parts[2].capitalize()} {parts[0][0].upper()}."
        return f"{parts[-1].capitalize()} {parts[0][0].upper()}."
    return slug.capitalize()


def _random_delay():
    """Delay aleatorio humanizado entre requests."""
    time.sleep(random.uniform(*REQUEST_DELAY))


def _validate_odds_pair(o1, o2):
    """
    Valida que un par de cuotas sea coherente:
    - Ambas > 1.01
    - Overround entre 98% y 115%
    """
    if o1 < 1.01 or o2 < 1.01 or o1 > 50 or o2 > 50:
        return False
    margin = (1/o1) + (1/o2)
    return 0.98 <= margin <= 1.15


# =====================================================================
# SCRAPER 1: ODDSPORTAL (fuente principal)
# =====================================================================

def scrape_oddsportal(page, tournament_url):
    """
    Scrapea partidos de OddsPortal usando Playwright.
    
    OddsPortal es una SPA (React/Next.js), las cuotas se cargan via JS.
    Playwright renderiza todo y luego parseamos con BeautifulSoup.
    
    Args:
        page: Playwright page object (ya creado)
        tournament_url: URL del torneo en OddsPortal
    
    Returns:
        Lista de dicts con {p1_name, p2_name, p1_odds, p2_odds, ...}
    """
    print(f"   [OddsPortal] Cargando {tournament_url}...")
    
    try:
        page.goto(tournament_url, timeout=PAGE_TIMEOUT, wait_until="networkidle")
    except Exception as e:
        print(f"   [OddsPortal] Error cargando página: {e}")
        # Intentar con domcontentloaded que es menos estricto
        try:
            page.goto(tournament_url, timeout=PAGE_TIMEOUT, wait_until="domcontentloaded")
            page.wait_for_timeout(5000)  # esperar JS extra
        except Exception as e2:
            print(f"   [OddsPortal] Fallo total: {e2}")
            return []
    
    # Esperar a que aparezcan los partidos (los contenedores de eventos)
    try:
        page.wait_for_selector("div.eventRow, div[class*='eventRow'], a[href*='/tennis/']", 
                               timeout=10000)
    except:
        print("   [OddsPortal] No se encontraron filas de partidos en el DOM")
    
    # Dar tiempo extra para cuotas
    page.wait_for_timeout(3000)
    
    # Opcional: cambiar a cuotas decimales si está en otro formato
    try:
        # Buscar selector de formato de cuotas
        odds_format = page.query_selector("button[data-testid='odds-format'], [class*='oddsFormat']")
        if odds_format:
            current = odds_format.inner_text()
            if "decimal" not in current.lower():
                odds_format.click()
                page.wait_for_timeout(1000)
                decimal_opt = page.query_selector("text=Decimal")
                if decimal_opt:
                    decimal_opt.click()
                    page.wait_for_timeout(2000)
    except:
        pass  # Si no encontramos el selector, probablemente ya está en decimal
    
    html = page.content()
    soup = BeautifulSoup(html, "lxml")
    
    return _parse_oddsportal_html(soup)


def _parse_oddsportal_html(soup):
    """
    Parsea el HTML renderizado de OddsPortal.
    
    Estructura real del DOM (verificada 2026-03-24):
    - Cada partido es un div.eventRow
    - Nombres: <p class="participant-name"> dentro de cada eventRow
    - Cuotas: <p data-testid="odd-container-default"> dentro de cada eventRow
    - URLs: <a href="/tennis/..."> dentro de cada eventRow
    - Fallback: JSON-LD <script type="application/ld+json"> en el <head>
    """
    matches = []
    
    # ── Estrategia 1: DOM directo con selectores específicos ──
    event_rows = soup.find_all("div", class_=re.compile(r'eventRow'))
    
    # Filtrar filas ocultas (OddsPortal inyecta honeypot rows con position:absolute)
    visible_rows = []
    for row in event_rows:
        style = row.get("style", "")
        if "position: absolute" in style or "left: -9999" in style:
            continue
        visible_rows.append(row)
    
    for row in visible_rows:
        # Extraer nombres de jugadores
        names = row.find_all("p", class_="participant-name")
        if len(names) < 2:
            continue
        
        p1_raw = names[0].get_text(strip=True)
        p2_raw = names[1].get_text(strip=True)
        
        # Filtrar dobles
        if "/" in p1_raw or "/" in p2_raw:
            continue
        
        # Extraer cuotas
        odds_elements = row.find_all("p", attrs={"data-testid": "odd-container-default"})
        odds_values = []
        for el in odds_elements:
            text = el.get_text(strip=True)
            try:
                val = float(text)
                if 1.01 <= val <= 50.0:
                    odds_values.append(val)
            except (ValueError, TypeError):
                pass
        
        if len(odds_values) < 2:
            continue
        
        p1_odds = odds_values[0]
        p2_odds = odds_values[1]
        
        if not _validate_odds_pair(p1_odds, p2_odds):
            continue
        
        # Extraer URL del partido
        match_link = row.find("a", href=re.compile(r'/tennis/'))
        href = match_link.get("href", "") if match_link else ""
        url = f"https://www.oddsportal.com{href}" if href.startswith("/") else href
        
        margin = (1/p1_odds) + (1/p2_odds)
        
        p1_name = normalize_player_name(p1_raw)
        p2_name = normalize_player_name(p2_raw)
        
        matches.append({
            "p1_name": p1_name,
            "p2_name": p2_name,
            "p1_name_raw": p1_raw,
            "p2_name_raw": p2_raw,
            "p1_odds": p1_odds,
            "p2_odds": p2_odds,
            "overround": round(margin, 4),
            "source": "oddsportal",
            "url": url,
        })
    
    if matches:
        return matches
    
    # ── Estrategia 2 (fallback): JSON-LD structured data ──
    # OddsPortal siempre incluye JSON-LD con "name": "Korda S. - Landaluce M."
    # Los nombres están ahí incluso si el DOM de cuotas no se parsea bien
    return _parse_oddsportal_jsonld(soup)


def _parse_oddsportal_jsonld(soup):
    """
    Fallback: extrae partidos del JSON-LD embebido en el HTML.
    OddsPortal incluye <script type="application/ld+json"> con SportsEvent
    que tiene "name": "Korda S. - Landaluce M." y URLs de partidos.
    
    Luego intenta asociar cuotas del DOM.
    """
    import json as json_mod
    
    matches = []
    scripts = soup.find_all("script", type="application/ld+json")
    
    events = []
    for script in scripts:
        try:
            data = json_mod.loads(script.string)
            # Puede ser un evento individual o un BreadcrumbList
            types = data.get("@type", [])
            if isinstance(types, str):
                types = [types]
            if "SportsEvent" in types:
                events.append(data)
        except:
            pass
    
    # Recoger todas las cuotas del DOM en orden
    all_odds = []
    for el in soup.find_all("p", attrs={"data-testid": "odd-container-default"}):
        try:
            val = float(el.get_text(strip=True))
            if 1.01 <= val <= 50.0:
                all_odds.append(val)
        except:
            pass
    
    for i, event in enumerate(events):
        name = event.get("name", "")
        url = event.get("url", "")
        
        # Nombre tiene formato "Korda S. - Landaluce M."
        parts = re.split(r'\s*-\s*', name, maxsplit=1)
        if len(parts) != 2:
            continue
        
        p1_raw = parts[0].strip()
        p2_raw = parts[1].strip()
        
        if "/" in p1_raw or "/" in p2_raw:
            continue
        
        # Intentar emparejar cuotas (2 por partido, en orden)
        odds_idx = i * 2
        if odds_idx + 1 < len(all_odds):
            p1_odds = all_odds[odds_idx]
            p2_odds = all_odds[odds_idx + 1]
            
            if not _validate_odds_pair(p1_odds, p2_odds):
                continue
            
            margin = (1/p1_odds) + (1/p2_odds)
        else:
            continue
        
        p1_name = normalize_player_name(p1_raw)
        p2_name = normalize_player_name(p2_raw)
        
        matches.append({
            "p1_name": p1_name,
            "p2_name": p2_name,
            "p1_name_raw": p1_raw,
            "p2_name_raw": p2_raw,
            "p1_odds": p1_odds,
            "p2_odds": p2_odds,
            "overround": round(margin, 4),
            "source": "oddsportal",
            "url": url if url else "",
        })
    
    return matches


# =====================================================================
# SCRAPER 2: ODDSCHECKER (fallback)
# =====================================================================

def scrape_oddschecker(page, tournament_url, tournament_slug):
    """
    Scrapea partidos de Oddschecker usando Playwright.
    
    Oddschecker es más agresivo contra bots, pero con un navegador real
    headless funciona razonablemente.
    """
    print(f"   [Oddschecker] Cargando {tournament_url}...")
    
    try:
        page.goto(tournament_url, timeout=PAGE_TIMEOUT, wait_until="domcontentloaded")
        page.wait_for_timeout(5000)  # JS pesado
    except Exception as e:
        print(f"   [Oddschecker] Error: {e}")
        return []
    
    # Aceptar cookies si aparece el banner
    try:
        accept_btn = page.query_selector("button[id*='accept'], button[class*='accept'], "
                                          "button:text('Accept'), button:text('Aceptar')")
        if accept_btn:
            accept_btn.click()
            page.wait_for_timeout(1000)
    except:
        pass
    
    html = page.content()
    soup = BeautifulSoup(html, "lxml")
    
    # Buscar enlaces a partidos individuales
    pattern = re.compile(rf'/es/tenis/{re.escape(tournament_slug)}/([a-z0-9-]+-v-[a-z0-9-]+)')
    match_links = soup.find_all("a", href=pattern)
    
    # Deduplicar
    seen_slugs = set()
    unique_matches = []
    for link in match_links:
        href = link.get("href", "")
        m = pattern.search(href)
        if m:
            slug = m.group(1)
            # Filtrar dobles (demasiados guiones o "and")
            if slug.count('-') < 7 and "and" not in slug and slug not in seen_slugs:
                seen_slugs.add(slug)
                unique_matches.append(slug)
    
    print(f"   [Oddschecker] {len(unique_matches)} partidos individuales encontrados")
    
    matches = []
    for match_slug in unique_matches:
        match_url = f"https://www.oddschecker.com/es/tenis/{tournament_slug}/{match_slug}"
        
        try:
            _random_delay()
            page.goto(match_url, timeout=PAGE_TIMEOUT, wait_until="domcontentloaded")
            page.wait_for_timeout(4000)  # esperar cuotas
            
            match_html = page.content()
            match_soup = BeautifulSoup(match_html, "lxml")
            
            parts = match_slug.split("-v-")
            if len(parts) != 2:
                continue
            
            p1_slug, p2_slug = parts[0], parts[1]
            p1_name = normalize_player_name(p1_slug)
            p2_name = normalize_player_name(p2_slug)
            
            result = _extract_oddschecker_odds(match_soup)
            
            if result:
                margin = (1/result["p1_odds"]) + (1/result["p2_odds"])
                matches.append({
                    "p1_name": p1_name,
                    "p2_name": p2_name,
                    "p1_name_raw": p1_slug,
                    "p2_name_raw": p2_slug,
                    "p1_odds": result["p1_odds"],
                    "p2_odds": result["p2_odds"],
                    "overround": round(margin, 4),
                    "source": "oddschecker",
                    "url": match_url,
                })
                print(f"      ✓ {p1_name} ({result['p1_odds']:.2f}) vs "
                      f"{p2_name} ({result['p2_odds']:.2f})")
            else:
                print(f"      ✗ Sin cuotas: {p1_name} vs {p2_name}")
                
        except Exception as e:
            print(f"      Error en {match_slug}: {e}")
    
    return matches


def _extract_oddschecker_odds(soup):
    """
    Extrae cuotas de una página de partido de Oddschecker.
    
    Con Playwright ya tenemos el HTML renderizado completo,
    así que las cuotas deberían estar presentes en el DOM.
    
    Busca múltiples estrategias:
    1. data-odig (atributo de cuota decimal)
    2. Elementos con clase que contenga "odds"
    3. Fallback: buscar pares numéricos coherentes
    """
    # Estrategia 1: data-odig (atributo específico de Oddschecker)
    odig_elements = soup.find_all(attrs={"data-odig": True})
    if odig_elements:
        odds_values = []
        for el in odig_elements:
            try:
                val = float(el["data-odig"])
                if 1.01 <= val <= 50.0:
                    odds_values.append(val)
            except:
                pass
        
        if len(odds_values) >= 2:
            # Las primeras dos cuotas suelen ser las del partido principal
            for i in range(len(odds_values)):
                for j in range(i+1, len(odds_values)):
                    if _validate_odds_pair(odds_values[i], odds_values[j]):
                        return {"p1_odds": odds_values[i], "p2_odds": odds_values[j]}
    
    # Estrategia 2: Buscar en la tabla de cuotas
    odds_cells = soup.find_all(["td", "span", "div"], 
                                class_=re.compile(r'odds|price|decimal', re.I))
    if odds_cells:
        odds_values = []
        for cell in odds_cells:
            text = cell.get_text(strip=True)
            try:
                val = float(text)
                if 1.01 <= val <= 50.0:
                    odds_values.append(val)
            except:
                # Intentar con fracciones (5/2)
                frac = re.match(r'^(\d+)/(\d+)$', text)
                if frac:
                    val = int(frac.group(1)) / int(frac.group(2)) + 1
                    if 1.01 <= val <= 50.0:
                        odds_values.append(round(val, 3))
        
        if len(odds_values) >= 2:
            for i in range(len(odds_values)):
                for j in range(i+1, min(i+20, len(odds_values))):
                    if _validate_odds_pair(odds_values[i], odds_values[j]):
                        return {"p1_odds": odds_values[i], "p2_odds": odds_values[j]}
    
    # Estrategia 3: Búsqueda de texto completa con overround
    full_text = soup.get_text(" ", strip=True)
    
    # Buscar decimales en contexto HTML (no versiones de scripts)
    # Usamos el HTML crudo para los patrones con contexto
    raw_html = str(soup)
    
    patterns = [
        r'data-odig="(\d+\.\d+)"',
        r'"oddsDecimal"\s*:\s*(\d+\.\d+)',
        r'data-o="(\d{1,3}/\d{1,3})"',  # fracciones
        r'>(\d+\.\d{2})<',               # texto entre tags
    ]
    
    all_odds = []
    for pattern in patterns:
        for m in re.finditer(pattern, raw_html):
            text = m.group(1)
            try:
                if "/" in text:
                    num, den = text.split("/")
                    val = int(num) / int(den) + 1
                else:
                    val = float(text)
                if 1.01 <= val <= 50.0:
                    all_odds.append(val)
            except:
                pass
    
    # Buscar el primer par coherente
    for i in range(len(all_odds)):
        for j in range(i+1, min(i+40, len(all_odds))):
            if _validate_odds_pair(all_odds[i], all_odds[j]):
                return {"p1_odds": all_odds[i], "p2_odds": all_odds[j]}
    
    return None


# =====================================================================
# INTERFAZ PRINCIPAL
# =====================================================================

def fetch_matches(tournament_slug, source="auto"):
    """
    Interfaz principal: obtiene partidos y cuotas de un torneo.
    
    Args:
        tournament_slug: clave del torneo (ej. "atp-miami") 
                        O una URL directa de OddsPortal (ej. "https://www.oddsportal.com/tennis/...")
        source: "oddsportal", "oddschecker", o "auto"
    
    Returns:
        Lista de dicts con partidos y cuotas, más metadata del torneo.
    """
    from playwright.sync_api import sync_playwright
    
    # ── Detectar si es una URL directa ──
    is_url = tournament_slug.startswith("http")
    
    if is_url:
        config = _build_config_from_url(tournament_slug)
    else:
        config = TOURNAMENT_CONFIG.get(tournament_slug)
    
    if not config:
        print(f"ERROR: Torneo '{tournament_slug}' no configurado")
        print(f"\nTorneos disponibles ({len(TOURNAMENT_CONFIG)}):")
        list_available_tournaments()
        print(f"\nO usa una URL directa de OddsPortal:")
        print(f"  python daily_scanner_v2.py --tournament=https://www.oddsportal.com/tennis/usa/atp-miami/")
        return [], {}
    
    print(f"\n{'─'*60}")
    print(f"  Scrapeando: {config['name']} ({tournament_slug})")
    print(f"  {config['surface']} | {config['court']} | {config['series']}")
    print(f"{'─'*60}")
    
    matches = []
    
    with sync_playwright() as pw:
        browser = pw.chromium.launch(
            headless=HEADLESS,
            slow_mo=SLOW_MO,
            args=[
                "--disable-blink-features=AutomationControlled",
                "--no-sandbox",
                "--disable-dev-shm-usage",
            ]
        )
        
        context = browser.new_context(
            viewport={"width": 1920, "height": 1080},
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36"
            ),
            locale="es-ES",
            timezone_id="Europe/Madrid",
        )
        
        # Inyectar anti-detección
        context.add_init_script("""
            Object.defineProperty(navigator, 'webdriver', {get: () => undefined});
            Object.defineProperty(navigator, 'plugins', {get: () => [1, 2, 3, 4, 5]});
            Object.defineProperty(navigator, 'languages', {get: () => ['es-ES', 'es', 'en']});
            window.chrome = { runtime: {} };
        """)
        
        page = context.new_page()
        
        # Intentar OddsPortal primero
        if source in ("auto", "oddsportal") and config.get("oddsportal"):
            try:
                matches = scrape_oddsportal(page, config["oddsportal"])
                if matches:
                    print(f"\n   [OddsPortal] ✓ {len(matches)} partidos obtenidos")
            except Exception as e:
                print(f"   [OddsPortal] Error: {e}")
        
        # Fallback a Oddschecker
        if not matches and source in ("auto", "oddschecker") and config.get("oddschecker"):
            print("   Intentando Oddschecker como fallback...")
            try:
                matches = scrape_oddschecker(page, config["oddschecker"], tournament_slug)
                if matches:
                    print(f"\n   [Oddschecker] ✓ {len(matches)} partidos obtenidos")
            except Exception as e:
                print(f"   [Oddschecker] Error: {e}")
        
        browser.close()
    
    # Añadir metadata del torneo a cada partido
    for m in matches:
        m["surface"] = config["surface"]
        m["court"] = config["court"]
        m["series"] = config["series"]
        m["tournament"] = config["name"]
    
    if not matches:
        print("\n   ⚠ No se pudieron obtener partidos de ninguna fuente")
        print("   Opciones:")
        print("   1. Usar --manual para introducir partidos a mano")
        print("   2. Verificar que el torneo esté en curso")
        print("   3. Ejecutar con HEADLESS=False para debug visual")
    
    return matches, config


def _build_config_from_url(url):
    """
    Construye config para una URL libre de OddsPortal.
    Pide al usuario los datos que faltan (superficie, serie).
    """
    print(f"\n  URL directa detectada: {url}")
    
    # Intentar inferir nombre del torneo de la URL
    # https://www.oddsportal.com/tennis/usa/atp-miami/ → "atp-miami"
    m = re.search(r'/tennis/[^/]+/([^/]+)', url)
    name = m.group(1).replace("-", " ").title() if m else "Torneo personalizado"
    
    # Inferir superficie del nombre
    surface_guess = "Hard"
    name_lower = name.lower()
    if any(c in name_lower for c in ["clay", "tierra"]):
        surface_guess = "Clay"
    elif any(g in name_lower for g in ["grass", "wimbledon", "halle", "queens", "stuttgart", "eastbourne", "mallorca", "hertogenbosch"]):
        surface_guess = "Grass"
    
    # Inferir serie
    series_guess = "ATP250"
    if any(s in name_lower for s in ["open", "masters", "slam"]):
        series_guess = "Masters 1000"
    
    print(f"  Nombre inferido: {name}")
    surface = input(f"  Superficie ({surface_guess}) [Hard/Clay/Grass]: ").strip() or surface_guess
    court = input(f"  Court (Outdoor) [Outdoor/Indoor]: ").strip() or "Outdoor"
    series = input(f"  Serie ({series_guess}) [Grand Slam/Masters 1000/ATP500/ATP250]: ").strip() or series_guess
    
    return {
        "surface": surface,
        "court": court,
        "series": series,
        "name": name,
        "oddsportal": url,
    }


def list_available_tournaments():
    """Muestra los torneos configurados, agrupados por serie."""
    by_series = defaultdict(list)
    for slug, cfg in sorted(TOURNAMENT_CONFIG.items()):
        by_series[cfg["series"]].append((slug, cfg))
    
    series_order = ["Grand Slam", "Masters 1000", "ATP500", "ATP250", "Masters Cup"]
    print(f"\nTorneos configurados ({len(TOURNAMENT_CONFIG)}):")
    for series in series_order:
        items = by_series.get(series, [])
        if not items:
            continue
        print(f"\n  ── {series} ({len(items)}) ──")
        for slug, cfg in items:
            print(f"    {slug:25s} {cfg['name']:25s} {cfg['surface']:6s} {cfg['court']}")
    
    print(f"\n  También puedes usar cualquier URL de OddsPortal directamente:")
    print(f"    --tournament=https://www.oddsportal.com/tennis/usa/atp-miami/")


# =====================================================================
# TEST / DEMO
# =====================================================================
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        list_available_tournaments()
        print("\nUso: python odds_scraper.py <torneo> [--source=oddsportal|oddschecker]")
        print("Ejemplo: python odds_scraper.py atp-miami")
        sys.exit(0)
    
    tournament = sys.argv[1]
    source = "auto"
    for arg in sys.argv[2:]:
        if arg.startswith("--source="):
            source = arg.split("=")[1]
    
    matches, config = fetch_matches(tournament, source=source)
    
    if matches:
        print(f"\n{'='*70}")
        print(f"  PARTIDOS ENCONTRADOS: {len(matches)}")
        print(f"{'='*70}")
        for m in matches:
            print(f"  {m['p1_name']:25s} @ {m['p1_odds']:.2f}  vs  "
                  f"{m['p2_name']:25s} @ {m['p2_odds']:.2f}  "
                  f"[margin: {m['overround']:.3f}] ({m['source']})")
    else:
        print("\nNo se encontraron partidos.")