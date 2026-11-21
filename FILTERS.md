# Filters and configuration examples

Edit **`config.toml`**, commit, then push the commit to **`deploy`** to change the
Pi. Pushing only to `main` does not change the display. The usual workflow keeps
both branches in sync:

```bash
git switch main
# Edit config.toml
python3 -m news_art.worker --check
git add config.toml
git commit -m "Change news filters"
git push origin main
git push origin main:deploy
```

The Pi checks for a release about every two minutes. An accepted update restarts
the worker, so it fetches a headline immediately rather than waiting another ten
minutes. If a query returns no recent stories, the last poster remains visible.

If editing on GitHub's website, editing `deploy` directly also works, but then
merge that change back into `main` before your next normal deployment. Working
on `main` and merging it into `deploy` avoids those branches drifting apart.

## Topics are free text

`news.query` is a Google News search, not a fixed category enum. There is no
complete list of allowed values: use topics, places, people, teams, events or
combinations. These are search examples, not guaranteed dedicated news sections.
Some will have few or no recent stories on a particular day.

Use `query = ""` for the general feed. `"all"` is a search word, not an all-topics
setting. The general feed contains Google-selected headlines; it is not every
article Google indexes and does not guarantee equal coverage of every topic.

Copy any example below into `query`, for example `query = "astronomy"`.

| Area | Example searches |
| --- | --- |
| Broad news | `world`, `US politics`, `international relations`, `elections`, `diplomacy` |
| Science | `science`, `astronomy`, `physics`, `biology`, `archaeology`, `paleontology` |
| Space | `space`, `NASA`, `ESA`, `Mars`, `Moon`, `James Webb telescope`, `SpaceX` |
| Technology | `technology`, `artificial intelligence`, `robotics`, `semiconductors`, `cybersecurity`, `quantum computing` |
| Gadgets and software | `smartphones`, `Linux`, `open source`, `Raspberry Pi`, `video games` |
| Environment | `climate`, `renewable energy`, `wildlife`, `conservation`, `oceans`, `biodiversity` |
| Business | `business`, `economy`, `inflation`, `trade`, `manufacturing`, `startups` |
| Transport | `electric vehicles`, `aviation`, `railways`, `public transport`, `shipping` |
| Health | `health`, `medical research`, `public health`, `nutrition`, `neuroscience` |
| Culture | `art`, `museums`, `architecture`, `photography`, `books`, `history` |
| Entertainment | `movies`, `music`, `television`, `film festivals`, `concerts` |
| Lifestyle | `travel`, `food`, `education`, `urban planning`, `design` |
| Sports | `sports`, `football`, `soccer`, `basketball`, `tennis`, `cycling`, `Formula 1`, `skiing`, `athletics` |
| Places | `Spain`, `Slovenia`, `United States`, `Europe`, `Japan`, `Africa`, `Latin America` |
| Cities | `Madrid`, `Barcelona`, `Ljubljana`, `London`, `New York`, `Tokyo` |
| Specific interests | `Real Madrid`, `Tour de France`, `ocean exploration`, `animal discoveries`, `space exploration` |

## Combine, broaden or exclude

The app passes your search to Google unchanged. Google News supports searches
for topics and places, exact phrases, included/excluded words and publisher sites.
RSS search behavior is best effort and can change; this app does not implement a
second local keyword filter.

| Goal | A complete TOML line to copy |
| --- | --- |
| General headlines | `query = ""` |
| Spanish news in the configured language | `query = "Spain"` |
| Spanish sports | `query = "sports Spain"` |
| Space news | `query = "space exploration"` |
| Broader mix of interests | `query = "science OR technology OR sports"` |
| Either of two countries | `query = "Spain OR Portugal"` |
| Exclude a word | `query = "technology -cryptocurrency"` |
| Exclude several words | `query = "sports -betting -gambling"` |
| Exact phrase | `query = '"climate change"'` |
| Phrase and a place | `query = '"renewable energy" Spain'` |
| Restrict to a publisher | `query = "site:bbc.com science"` |
| Restrict to a publisher and place | `query = "site:reuters.com Spain"` |

Use uppercase `OR`. Plain words together narrow the search; `sports Spain`
asks for coverage matching both concepts, whereas `sports OR Spain` broadens it.
Single quotes around the TOML value let you include double quotes in a phrase
without escaping them. Keep `site:` attached to its domain.

Restricting the publisher still uses the same Google News RSS provider. No new
feed integration, API key or publisher subscription is added. Articles themselves
may be paywalled; the app only displays the RSS headline and attribution.

Google's references: [News search](https://support.google.com/googlenews/answer/9005601)
and [search operators](https://support.google.com/websearch/answer/2466433).
These describe Google search features, not a guaranteed RSS API contract. The
general feed and `sports Spain` search were checked live during project setup.

## Language and edition

Keep these defaults for English US-edition and international coverage:

```toml
language = "en-US"
country = "US"
```

`country` chooses the feed edition. It does **not** restrict every story to that
country. To request Spain-related English news, keep the defaults and set
`query = "Spain"`. There is no separate country-of-event filter in this version.

For an English UK edition, use `language = "en-GB"` and `country = "GB"`.
For a Spanish Spain edition, use `language = "es"` and `country = "ES"`.
The app constructs Google's edition identifier from these fields; not every
arbitrary language/country combination is available from Google.

## Freshness and refresh interval

`max_age_hours` rejects articles whose RSS publication date is older than that
many hours. It does not set how often the screen changes or establish when the
event itself occurred. Allowed range: **1–168 hours**.

| Value | Use |
| --- | --- |
| `12` | Very recent stories; fewer matches for narrow topics |
| `24` | Last day |
| `48` | Default: last two days |
| `72` | More options for quieter topics |
| `168` | Last week |

`display.refresh_seconds` controls the interval between successful updates:

| Value | Interval |
| --- | --- |
| `60` | One minute; useful briefly when checking rotation |
| `300` | Five minutes |
| `600` | Ten minutes; default |
| `1800` | Thirty minutes |
| `3600` | One hour |

Allowed range: **30–86400 seconds**. Failed news fetches retry after one minute
while retaining the last poster. Recent unseen stories are preferred; stories can
repeat once the current feed's available stories are exhausted. The three demo
images rotate independently of headline selection.

## Art instructions are also free text

Edit `generation.style`. There are no preset names to memorize. Example values:

```toml
style = "A realistic editorial illustration, natural lighting, detailed composition."
# Other examples; enable only one style line:
# style = "An expressive oil painting in the style of Picasso."
# style = "A soft watercolor illustration with warm colors."
# style = "A vintage travel poster, bold shapes and a limited color palette."
# style = "A humorous newspaper cartoon, playful and clear."
# style = "A surreal dreamlike scene with unexpected visual metaphors."
# style = "A cinematic science-fiction scene with dramatic lighting."
# style = "A black-and-white charcoal drawing with strong contrast."
# style = "A detailed clay miniature photographed with shallow depth of field."
```

**Currently this only changes the saved prompt, not the three demo images.** The
future model backend will receive this instruction followed by the selected
headline and the instruction `No text or lettering.` Model results will depend
on its capabilities; writing a style is not a guarantee it can reproduce it.

Leave `backend = "demo"` until another backend is implemented. The only supported
backend today is `demo`. Leave `image_size = 256` for the planned Supra2 baseline.
The demo backend accepts 64–1024 pixels but that does not establish the future
model's supported resolutions. Originals in `imgs/` are preserved.

`display.title` is free text for the small heading above the poster.
