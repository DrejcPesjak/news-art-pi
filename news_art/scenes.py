"""Research baseline: offline topic-to-scene rules, not an article summarizer.

Known topics become generic visual illustrations. They intentionally omit names,
scores, quotations and other details the image model tends to render as lettering.
Unknown topics keep the cleaned headline rather than inventing a scene.
Production v3 uses brief_scenes instead; these rules remain for old comparisons.
"""
import re


# Specific events precede broad topics: a ship fire is not a generic war scene,
# and a cancer drug study is not a celebrity portrait. Patterns use word
# boundaries so e.g. "AI" cannot accidentally match "said" or "Spain".
RULES = [
    ("space-mercury", r"\bbepicolombo\b|\bmercury\b.*\b(orbit|probe|planet)\b",
     "A spacecraft with solar panels orbiting the cratered planet Mercury, the curved rocky planet below, dark outer space and distant stars."),
    ("space-return", r"\b(capsule|spacecraft|dragon)\b.*\b(home|return|earth)\b",
     "A space capsule descending above the curved blue Earth, white clouds and the dark sky of outer space."),
    ("space-launch", r"\b(rocket|spacecraft|satellite)\b.*\b(launch|launches|launching|liftoff)\b|\b(launch|launches|launching)\b.*\b(rocket|satellite)\b",
     "A rocket rising from a launch pad, bright orange flames and billowing white smoke beneath it, blue sky above."),
    ("space", r"\b(astronaut|spacecraft|space station|telescope|exoplanet)\b",
     "A spacecraft floating above a rocky planet in outer space, distant stars and sunlight along the planet's curved horizon."),
    ("ship-fire", r"\b(ship|vessel|tanker)\b.*\b(ablaze|fire|burning|strike)\b",
     "A cargo ship on the open sea with orange flames and dark smoke rising from its deck, distant view across the water."),
    ("flood", r"\b(flood|flooding|floods|tides?|king tide)\b",
     "Seawater covering a coastal street, buildings and palm trees reflected in the water, an overcast sky."),
    ("wildfire", r"\b(wildfire|wildfires|forest fire)\b",
     "Orange flames burning through a forest on a hillside, thick smoke drifting above the trees, a distant landscape view."),
    ("storm", r"\b(hurricane|tornado|storm|typhoon|cyclone)\b",
     "A powerful storm approaching a coastal town, dark swirling clouds, strong wind bending trees and waves crashing onto the shore."),
    ("earthquake", r"\b(earthquake|quake)\b",
     "A damaged city street after an earthquake, cracked pavement, fallen bricks and emergency workers in the distance."),
    ("wedding", r"\b(wedding|marries|married|marry|bride|groom)\b",
     "A bride in a white wedding dress and a groom in a dark suit standing in a garden, flowers and warm afternoon sunlight."),
    ("soccer", r"\b(nwsl|soccer|premier league|la liga|champions league|fifa)\b",
     "Women soccer players on a green soccer field, a soccer ball on the grass, a stadium crowd in the background."),
    ("american-football", r"\b(nfl|quarterback|qb|touchdowns?|td runs?|cowboys|colts)\b|\b(tennessee|ucla|nebraska)\b.*\b(lsu|michigan|ohio state)\b",
     "American football players wearing helmets and shoulder pads on a green football field, one player carrying an oval football, a stadium crowd behind them."),
    ("basketball", r"\b(nba|wnba|basketball)\b",
     "Basketball players on an indoor wooden court, an orange basketball near the hoop, a crowd in the arena."),
    ("combat-sports", r"\b(ufc|mma|boxing|boxer)\b",
     "Two athletes wearing boxing gloves facing each other inside a sports ring, bright arena lights and a distant crowd."),
    ("swimming", r"\b(swims?|swimming|swimmer)\b",
     "A swimmer moving through blue ocean water, small waves and a distant coastline under a clear sky."),
    ("video-games", r"\b(gta|grand theft auto|video game|gaming|playstation|xbox)\b",
     "A video game controller on a wooden desk, a glowing computer screen showing a colorful city street, soft room lighting."),
    ("movies", r"\b(movie|film|cinema|box office|hunger games)\b",
     "Rows of red seats inside a cinema, a large glowing screen showing a mountain landscape, dim warm lighting."),
    ("music", r"\b(singer|concert|album|musician|tour dates)\b",
     "A singer holding a microphone on a concert stage, colorful spotlights and a crowd in the foreground."),
    ("fashion", r"\b(hair|hairstyle|fashion|outfit|dress)\b",
     "A woman with long brown hair wearing an elegant jacket, standing on a sunlit city street, soft afternoon light."),
    ("crime", r"\b(arrested|shoot-out|drug lord|wanted list|detained)\b",
     "A police car parked on a city street, blue emergency lights reflected on the road, buildings in the background."),
    ("medical-research", r"\b(drug|treatment|vaccine)\b.*\b(study|tumor|trial|cancer|research)\b|\b(tumor|clinical trial)\b",
     "A scientist wearing a white lab coat examining a sample through a microscope in a bright medical laboratory, glass sample vials on the bench."),
    ("health", r"\b(cancer|measles|hospital|healthcare|trumpcare|healing|disease)\b",
     "A doctor in a white coat speaking with a seated patient in a bright hospital room, soft daylight through a window."),
    ("chips", r"\b(nvidia|intel|semiconductor|processor|gpu|cpu|chipmaker)\b",
     "A small computer processor resting on a detailed green circuit board, metallic contacts and tiny electronic components, close-up view."),
    ("earbuds", r"\b(airpods|earbuds|headphones)\b",
     "A pair of white wireless earbuds beside an open charging case on a wooden table, soft window light."),
    ("phones", r"\b(iphone|smartphone|mobile|lock screen)\b",
     "A smartphone resting on a wooden desk, its screen filled with a colorful mountain landscape, soft window light."),
    ("cybersecurity", r"\b(hack|hacked|cyberattack|security flaws|outages|surveillance|screenshots)\b",
     "Rows of computer servers in a dark data center, small blue indicator lights, cables and metal cabinets."),
    ("ai", r"\b(ai|artificial intelligence|robots?)\b",
     "A small robot arm beside a computer on a laboratory workbench, metallic machinery and soft blue lighting."),
    ("finance", r"\b(s&p|stock market|stocks|investors|credit card|interest rates|inflation|economy|economic)\b",
     "A small stack of coins beside a shopping basket on a wooden table, a city skyline softly blurred in the background."),
    ("environment", r"\b(climate|cop\d+|fossil fuels|pollution|toxic|explosives|roundup)\b",
     "An industrial factory beside a green field, smoke rising from tall chimneys, a wide landscape beneath an overcast sky."),
    ("military", r"\b(pentagon|military|army|missiles?|war|troops|drone strike)\b",
     "Military vehicles parked on an open field, a distant row of buildings, a wide view under a cloudy sky."),
    ("politics", r"\b(trump|congress|senate|senator|democrats?|republicans?|government|governor|diplomats?|brexit|judicial|election)\b",
     "A group of officials in formal suits seated around a large wooden table in a government meeting room, tall windows and daylight."),
]
_COMPILED = [(name, re.compile(pattern, re.I), scene) for name, pattern, scene in RULES]


def scene_for(headline):
    """Return (rule name, visual caption); fallback preserves an unknown subject."""
    cleaned = re.sub(r"^(?:exclusive|breaking(?: news)?|live updates?)\s*[:|—–-]\s*", "", headline, flags=re.I)
    cleaned = re.sub(r"\s*[–—|]\s*(?:live reaction|as it happened|live updates).*$", "", cleaned, flags=re.I)
    cleaned = " ".join(cleaned.split()).strip()
    for name, pattern, scene in _COMPILED:
        if pattern.search(cleaned):
            if name == "soccer" and re.search(r"\b(win|wins|trophy|championship|champions|capture|stun)\b", cleaned, re.I):
                scene = "Women soccer players celebrating on a green soccer field, a player raising a silver trophy, a stadium crowd in the background."
            if name == "soccer" and not re.search(r"\b(nwsl|women|wnsl|wsl)\b", cleaned, re.I):
                scene = scene.replace("Women soccer players", "Soccer players")
            return name, scene
    return "headline-fallback", cleaned.rstrip(".!?") + "."
