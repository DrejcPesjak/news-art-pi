# News Art Pi

A self-contained Raspberry Pi news display. One Google News RSS feed, a new
headline every ten minutes, and a fullscreen HDMI poster. No API keys, paid
services, browser, or laptop needed at runtime.

The first release rotates three supplied demo images. They are unrelated to the
headlines and clearly labelled. The generator produces 256 × 256 RGB images,
preserving the original artwork with padding. The original files stay in `imgs/`.
The production display still uses demo images. Laptop Supra2 benchmarks and ARM
runtime experiments are documented in [research/README.md](research/README.md).

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
backend = "demo"
image_size = 256
style = "A realistic editorial illustration, natural lighting, detailed composition."

[display]
refresh_seconds = 600
title = "THE WORLD, REIMAGINED"
```

Google News is the only RSS provider; it aggregates multiple publishers. A query
selects one search feed instead of the general feed. Query filters are best effort,
and Google does not guarantee this RSS endpoint as a stable API. Source names and
story links are retained. No article scraping or full-text downloading is used.
The empty query covers general US-edition and international news. Leave language
as English when searching Spain if you want English headlines.

`style` is included in each stored prompt, ready for the future model backend.
It does not change the demo artwork. All three images cycle in sorted filename
order; story selection favors unseen recent headlines. Once the feed is exhausted,
stories can repeat. Failed refreshes keep the last poster and retry after a minute.

## Local development

Requires Python 3.10+ and Pillow (plus `tomli` on Python 3.10).
On Ubuntu/Raspberry Pi OS:

```bash
sudo apt-get install python3-pil python3-pil.imagetk python3-tk
# Python 3.10 only: sudo apt-get install python3-tomli
python3 -m unittest discover -s tests -v
python3 -m news_art.worker --check
python3 -m news_art.worker --once
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
active display. Configuration changes therefore take effect within about two
minutes. No GitHub login or deploy key is needed on the Pi. The laptop can be off.
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
sudo bash deployment/install.sh "$USER" https://github.com/DrejcPesjak/news-art-pi.git
```

On the next desktop login the viewer starts automatically. To start it immediately,
run `bash /opt/news-art-pi/current/deployment/start-display.sh` **in a desktop
terminal**. News fetching starts at boot independently of the desktop. Connecting
a different HDMI monitor needs no config change; reboot if the desktop does not
pick up its resolution. Keep using a supply and cable that report no undervoltage.

## Diagnostics and rollback

```bash
systemctl status news-art.service news-art-update.timer
journalctl -u news-art.service -n 40 --no-pager
journalctl -u news-art-update.service -n 40 --no-pager
systemctl --user status news-art-display.service
cat /opt/news-art-pi/current/REVISION
cat /var/lib/news-art-pi/state.json
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
- `news_art/generator.py`: demo backend and configurable prompt builder.
- `news_art/worker.py`: schedule, history, atomic image/state publication.
- `news_art/display.py`: lightweight fullscreen viewer.
- `deployment/`: initial installation, systemd units, GitHub update/rollback.
- `tests/`: parsing, filters, image rotation, persistence and failure handling.

Demo images are user-supplied reference assets; no license or ownership claim is
made for their artwork. They are not photographs of the displayed news events.
