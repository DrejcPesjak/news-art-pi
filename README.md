# News Art Pi

A self-contained Raspberry Pi news display. One Google News RSS feed, a new
headline every ten minutes, and a fullscreen HDMI poster. No API keys, paid
services, browser, or laptop needed at runtime.

The Pi generates 256 × 256 illustrations locally with Supra2, a Flan-T5 text
encoder, and the compatible distilled TAESD decoder. Models load sequentially
through NCNN to fit the Pi 2 B's RAM. A measured 25-step run took 9 min 30 s;
[research results](research/RESULTS.md) include quality comparisons and limits.
The last successful poster stays visible while the next image is generated.

## Change topics or art style

See [FILTERS.md](FILTERS.md) for topic examples, query syntax, language/edition
settings, art instructions, and the complete configuration reference.

Edit `config.toml`:

```toml
[news]
query = ""                # General news; or "sports", "Spain", "sports Spain"
language = "en-US"
country = "US"            # Feed edition, not a strict country-of-story filter
max_age_hours = 48

[generation]
backend = "supra"
image_size = 256
style = "oil painting"
steps = 25
guidance = 3.0
seed = 0
threads = 4
memory_mib = 480
timeout_seconds = 900

[display]
refresh_seconds = 600
title = "THE WORLD, REIMAGINED"
```

Google News is the only RSS provider; it aggregates multiple publishers. A query
selects one search feed instead of the general feed. Query filters are best effort,
and Google does not guarantee this RSS endpoint as a stable API. Source names and
story links are retained. For the selected article, the Pi makes a best-effort
publisher request for its description and opening paragraph; blocked or unavailable
pages fall back to the headline. Google redirect resolution is undocumented and
can break. Full articles are not archived.
The empty query covers general US-edition and international news. Leave language
as English when searching Spain if you want English headlines.

The v3 prompt builder uses local NLTK POS/NER and WordNet to prune source text
to one or two concrete subjects/settings, stripping identifying names. It starts
with `a` and appends `style`: `a Russian cargo ship hit by a drone. oil painting`.
Nationality adjectives may survive. This is a short illustration caption, not a
faithful article summary; roles, actions and context can be lost. If no concrete
subject is found it uses `a scene`. No extra generative model or paid API is used.
Caption preparation runs in a separate process that exits before NCNN starts.
The original headline remains on the display and in saved metadata.
The saved state retains its exact prompt, seed, and `story.caption` diagnostics,
including publisher context or fetch failure. Seeds advance with each poster.
Story selection favors unseen recent headlines; stories can repeat once the feed
is exhausted. Network or generation failures keep the previous image and retry
in a minute. Jobs never overlap. The ten-minute interval is measured from job
start, so a 9½-minute generation does not become a 19½-minute refresh.

Set `backend = "demo"` to rotate the three original pictures instead. Originals
remain in `imgs/`; only the demo backend pads them into equal-sized squares.

## Local development

Requires Python 3.10+ and Pillow (plus `tomli` on Python 3.10).
On Ubuntu/Raspberry Pi OS:

```bash
sudo apt-get install python3-pil python3-pil.imagetk python3-tk
# Python 3.10 only: sudo apt-get install python3-tomli
python3 -m unittest discover -s tests -v
python3 -m news_art.worker --check
# For a laptop display preview, copy config.toml to config.local.toml and set backend = "demo".
python3 -m news_art.worker --config config.local.toml --once
python3 -m news_art.display --windowed
```

Run `python3 -m news_art.worker` to refresh continuously. The viewer follows
`state/state.json`. F11 toggles fullscreen; Escape leaves fullscreen. Close the
window to return to the desktop. The layout adapts to monitor resolution.

## GitHub deployment

`main` is the development branch; **`deploy` is what the Pi runs**. Normal workflow:

**Pushing only to `main` does not update the Pi.** Publish the tested `main` commit
to `deploy` with `git push origin main:deploy`. Editing `deploy` directly also works,
but merge those changes back into `main` before the next normal deployment.

```bash
git switch main
# Edit, then:
python3 -m unittest discover -s tests -v
python3 -m news_art.worker --check
git add .
git commit -m "Change news filters"
git push origin main
git push origin main:deploy
```

The Pi polls the public repository over HTTPS every two minutes, validates the
candidate on-device, atomically switches releases, and restarts the worker and
active display. An accepted update begins a new generation within about two minutes. Its image
appears when generation finishes; the previous poster stays visible meanwhile. No GitHub login or deploy key is needed on the Pi. The laptop can be off.
All code pushed to `deploy` is trusted executable code on the Pi: treat that branch
as production. Run the local checks before pushing; the updater repeats the tests
and asset/configuration validation on the Pi before activating every release.
GitHub Actions is not configured: the laptop's GitHub login lacks the workflow
permission needed to upload workflow files.

Configuration and assets update automatically. Changes to systemd units, OS
packages or the installer need a deliberate maintenance installation; they are not
silently applied by the updater. Model weights must stay outside Git.

## First Pi installation

Requires Raspberry Pi OS desktop (32-bit is supported), network access and desktop
autologin. The current Pi already has autologin enabled. Installation changes no
passwords and does not require a GitHub account on the Pi:

```bash
git clone --branch deploy https://github.com/DrejcPesjak/news-art-pi.git
cd news-art-pi
# First install the native runtime and a verified model bundle (see below),
# or publish a backend = "demo" configuration to deploy first.
sudo bash deployment/install.sh "$USER" https://github.com/DrejcPesjak/news-art-pi.git
```

On the next desktop login the viewer starts automatically. To start it immediately,
run `bash /opt/news-art-pi/current/deployment/start-display.sh` **in a desktop
terminal**. News fetching starts at boot independently of the desktop. Connecting
a different HDMI monitor needs no config change; reboot if the desktop does not
pick up its resolution. Keep using a supply and cable that report no undervoltage.

## Native runtime and model installation

The current Pi is bootstrapped already. Its production models live under
`/var/lib/news-art-pi/models/supra` (an atomic link to a verified version), and the
NCNN Python package lives in `/opt/news-art-pi/runtime`. These paths work with the
worker service's protected filesystem. PyTorch and the laptop are not needed.

For a fresh Pi, build a portable bundle using [the research commands](research/README.md),
transfer it plus an NCNN wheel installed into a directory for the target's Python
and ARM architecture, and run once:

```bash
sudo bash deployment/install-models.sh "$USER" /path/to/verified-bundle /path/to/ncnn-runtime
sudo bash deployment/install-captions.sh "$USER" /path/to/extracted-nltk-data
```

This verifies checksums, installs system NumPy/SentencePiece, and copies model
assets into a versioned service-readable directory. Model weights stay outside
Git. Routine code, filters, style, and sampling changes only need a push to
`deploy`; package or model replacements require this maintenance bootstrap.
Caption bootstrap installs system NLTK/lxml and copies the private `chunkers`,
`corpora`, and `taggers` resources to `/var/lib/news-art-pi/models/nltk`.
Required datasets: `averaged_perceptron_tagger_eng`, `maxent_ne_chunker_tab`,
`words`, and `wordnet`; no runtime downloads are performed. A laptop may use its
existing `~/.cache/news-art-nltk`, or set `NEWS_ART_NLTK_DATA` explicitly.

`python3 -m news_art.worker --check` is an offline configuration/demo-asset check
that works on the laptop. `--check --check-runtime` additionally checks production
model files and native imports on the Pi without generating an image. The updater
runs this target check before activating future releases.

The production process uses a 480 MiB Linux allocation limit per inference stage
and a 900-second overall timeout. On this Pi, memory cgroups are unavailable;
this is not an exact RSS limit. Timeouts terminate the entire process group.

## Diagnostics and rollback

```bash
systemctl status news-art.service news-art-update.timer
journalctl -u news-art.service -n 40 --no-pager
journalctl -u news-art-update.service -n 40 --no-pager
systemctl --user status news-art-display.service
cat /opt/news-art-pi/current/REVISION
cat /var/lib/news-art-pi/state.json
cat /var/lib/news-art-pi/job.json
vcgencmd get_throttled
```

For a normal rollback, revert the bad commit on `main`, then push `main:deploy`.
Failed validation leaves the old release running; a detected worker startup or
active-display startup failure switches back. This is a basic startup check,
not a guarantee against every later runtime or display failure. Previous releases
remain under `/opt/news-art-pi/releases/` for emergency SSH recovery.

To stop automatic deployment: `sudo systemctl stop news-art-update.timer`.
To apply a change immediately: `sudo systemctl start news-art-update.service`.

## Structure

- `news_art/news.py`: build and parse the single RSS feed.
- `news_art/generator.py`: selectable demo/Supra backend and prompt builder.
- `news_art/brief_scenes.py`: v3 name-stripped short-caption converter.
- `news_art/captions.py`: isolated caption process and runtime validation.
- `news_art/article.py`: best-effort publisher description/opening extraction.
- `news_art/supra.py`: bounded native-runtime adapter and production asset checks.
- `news_art/worker.py`: schedule, history, atomic image/state publication.
- `news_art/display.py`: lightweight fullscreen viewer.
- `deployment/`: initial installation, systemd units, GitHub update/rollback.
- `tests/`: parsing, filters, image rotation, persistence and failure handling.

Demo images are user-supplied reference assets; no license or ownership claim is
made for their artwork. They are not photographs of the displayed news events.
