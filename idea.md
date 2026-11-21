**Yes, the idea is good.** A Raspberry Pi displaying AI-generated interpretations of live news every 10 minutes would make a pretty cool piece of office decoration. And using Codex to experiment with model compression makes it a much more interesting ML project than just building a slideshow.

But there's one significant challenge: **the Raspberry Pi 2 B is extremely underpowered for modern image generation**, even with a tiny model.

The good news is that you can build the entire system around your existing Pi, develop everything remotely from your laptop, and experiment with getting the model to run locally.

## **1. What I'd build**

**Live news feeds**

RSS / API · Every 10 minutes

**Prompt builder**

Summarize headline + randomly choose artistic instructions

"Draw this as a funny Christmas cartoon"

**Tiny image-generation model**

Raspberry Pi CPU, after optimization

Or laptop GPU as a fallback

**Office HDMI screen**

Fullscreen image + headline + timestamp

I'd design it so that image generation can run either locally on the Pi or remotely on your laptop without changing the news-fetching, scheduling, or display code.

That gives you a working decoration even if the local model turns out to be impractically slow.

## **2. The model we discussed: Supra2-IMG**

[SupraLabs/Supra2-IMG · Hugging Face](https://images.openai.com/static-rsc-4/PbWdP2RJRWpkfiKjxsLHtTdNN5B3ECZwzYuahlEJoB6SvaFk20NEoIDzV4MHbe4cjsunROuyuuN-dT6k9Tuz-gUr1YnnrWWaQYPIjCG2RGC05jo26Mo-u8idhjal3rKEvl2YZ0d7nFtzO2v9m_uN_bRBgeU5PDJMM6k8FSM9uVA?purpose=inline)

**Supra2-IMG**

104.1M-parameter diffusion transformer, generating 256 × 256 images.

[Hugging Face model and code](https://huggingface.co/SupraLabs/Supra2-IMG?utm_source=chatgpt.com)

This is the model we previously discussed. It really is small by image-generation standards, but the complete inference pipeline also requires a text encoder and image decoder.&#x20;

[image](https://www.google.com/s2/favicons?domain=https://huggingface.co\&sz=32)

Hugging Face+1



Here's the hardware problem:

| Component        | Your Raspberry Pi 2 B                     |
| ---------------- | ----------------------------------------- |
| CPU              | 4 cores, approximately 900 MHz            |
| RAM              | 1 GB                                      |
| AI accelerator   | None                                      |
| GPU              | Not practically usable for this model     |
| Storage          | 64 GB microSD (likely)                    |
| Operating system | Needs an ARM-compatible inference runtime |

The model's weights may fit after aggressive quantization, but **fitting in RAM does not mean generating images quickly**.

I'd make the local goal experimental: getting one valid image generated reliably. Only then would I optimize for the 10-minute refresh interval.

## **3. Can you increase the power? (Question 2)**

Yes, and you should fix the power warning before doing anything computationally intensive.

Your Raspberry Pi 2 B needs a reliable **5V, 2.5A Micro-USB power supply** (12.5W). That's Raspberry Pi's official recommendation.&#x20;

[image](https://www.google.com/s2/favicons?domain=https://www.raspberrypi.com\&sz=32)

Raspberry Pi+1



[Raspberry Pi PSU 5.1V 2.5A MicroUSB - EU-plug | Kiwi Electronics](https://images.openai.com/static-rsc-4/qjqnHwt-0ZxXus3jtZGlskdLZj1S93J-fyU4JkU6k8hVbs-eY8ogVkzqk9pOOtetq0ow8knSUCsbgfMNlNuGjdezAN3qCX6nc5TeWfHdYksck7hat0dF0G3wx6CFIyP9oMf4aHpCeQC1_ZQz0Vuoh3yMZmbSKDsqczbPuTOtIlI?purpose=inline)

**Recommended power supply**

5V / 2.5A, good-quality Micro-USB cable. A supply with a higher current rating is fine, provided it delivers the correct voltage.

Your existing 7-inch touchscreen is probably contributing to the issue. The original official display typically draws another 200mA at 5V, and it may be taking power directly from the Pi's GPIO pins.&#x20;

[image](https://www.google.com/s2/favicons?domain=https://www.raspberrypi.com\&sz=32)

Raspberry Pi Documentation



Switching to the larger, independently powered HDMI monitor should help.

After installing Raspberry Pi OS, you can test with:

```
vcgencmd get_throttled
```

`throttled=0x0` means no undervoltage or throttling events have been recorded since boot.

**Important distinction:** Giving the Pi a better power supply fixes instability and possible throttling. It doesn't upgrade the CPU or increase RAM. Modest overclocking may be possible, but it won't transform the image-generation performance. Don't feed it 9V or 12V.

## **4. Can Codex do everything? (Question 3)**

Almost all the software work, yes.

Codex can write the application, test it on your laptop, SSH into the Pi, deploy code, install packages, set up systemd services, run experiments, and collect benchmarks, provided it has the relevant permissions.

I'd **run Codex on your laptop**, not on the Pi. The Pi 2's old CPU architecture makes running the modern Codex environment directly on it an unnecessary complication.

### **4.1–4.3 Connecting your laptop to the Pi**

**Option A — Both on the router**

Best for permanent installation

Laptop

Wi-Fi

Router

Ethernet

Pi 2

Both machines are on the same local network. The Pi gets internet access, and your laptop accesses it using SSH.

The Pi 2 B has no built-in Wi-Fi, so Ethernet is the simplest connection. An office network may isolate wired and wireless clients, in which case SSH could be blocked.

**Option B — Ethernet directly between laptop and Pi**

Best for initial development

Ubuntu laptop

Direct Ethernet cable

Raspberry Pi

**Yes, this works!** No router, switch, special crossover cable, or extra hardware is required, assuming your laptop has an Ethernet port.

Since your laptop runs Ubuntu, you can set the wired connection's IPv4 mode to **Shared to other computers** in NetworkManager. That can also share your laptop's Wi-Fi internet access with the Pi.

**Option C — USB directly between laptop and Pi**

Not useful for network connectivity on the Pi 2 B. Its Micro-USB connection is for power, not USB-gadget networking.

### **How Codex actually connects**

Once you've installed Raspberry Pi OS and enabled SSH, from your laptop:

```
ssh youruser@raspberrypi.local
```

Replace `raspberrypi` with the hostname configured during setup. If `.local` discovery doesn't work, use the Pi's IP address.

You can then add a convenient alias to `~/.ssh/config`:

```
Host news-pi
    HostName 192.168.1.50
    User youruser
    IdentityFile ~/.ssh/id_ed25519
```

The IP is only an example; use the address assigned to your Pi.

After that:

```
ssh news-pi
```

Codex running on your laptop can use the same SSH connection to execute commands remotely, subject to its sandbox and approval settings. Cloud-hosted agents cannot automatically reach a private office LAN.

## **5. GitHub project + automatic pull (Question 3.3)**

Yes. This is how I'd organize it.

**Laptop + Codex**

Development, experiments, model optimization, tests

**GitHub repository**

Application code, configuration, documentation, deployment scripts

**Raspberry Pi**

Poll for approved updates, deploy, restart application automatically

I'd keep the repository roughly like this:

```
news-art-pi/
├── app/
│   ├── news.py
│   ├── prompts.py
│   ├── generator.py
│   ├── display.py
│   └── main.py
├── models/
│   ├── supra/
│   ├── optimize.py
│   └── benchmark.py
├── deployment/
│   ├── news-art.service
│   ├── update.service
│   └── update.timer
├── tests/
├── config.yaml
└── README.md
```

Use `systemd` to launch the application on boot and to check GitHub periodically. Once a tested update is available, it can be deployed and the service restarted.

For the early stages, direct SSH deployment is easier than waiting for GitHub pulls. Later, I'd automate deployments but use versioned releases or a deployment branch rather than blindly executing every change Codex pushes to `main`.

Keep model checkpoints outside Git, with downloadable versioned artifacts. A private repository can use a read-only deploy key.

## **6. Should you flash a new OS? (Question 4)**

**Yes. I'd replace Kali with Raspberry Pi OS Lite, 32-bit.**

Kali's extra security tools aren't useful here, and your installation is very old.

Because you have the physical Pi, keyboard, and monitor working already, there's no need to recover the old installation.

### **Kali's default credentials**

| Kali version                          | Username | Password |
| ------------------------------------- | -------- | -------- |
| Before 2020.1                         | `root`   | `toor`   |
| 2020.1 and later preconfigured images | `kali`   | `kali`   |

Given that you last used it around seven years ago, try `root` / `toor` first. It only works if the default password wasn't changed.&#x20;

[image](https://www.google.com/s2/favicons?domain=https://www.kali.org\&sz=32)

Kali Linux Documentation+1



### **New OS installation**

1. Remove the microSD card and insert it into your laptop using a card reader.
2. Install [Raspberry Pi Imager](https://www.raspberrypi.com/software/?utm_source=chatgpt.com).
3. Select Raspberry Pi 2, then **Raspberry Pi OS Lite (32-bit)**.
4. Configure a username, password, SSH access, and hostname such as `news-pi`.
5. Flash the card, insert it back into the Pi, connect Ethernet, and boot.

The modern Raspberry Pi OS is based on Debian Trixie, and the 32-bit version supports older Pi models.&#x20;

[image](https://www.google.com/s2/favicons?domain=https://www.raspberrypi.com\&sz=32)

Raspberry Pi+1



Lite doesn't come with a graphical desktop. That's intentional: with only 1GB of RAM, we want as little background overhead as possible. Codex can install a minimal X11 display environment and fullscreen image viewer.

For an HDMI image slideshow you don't need Chromium, a complete desktop, or a large browser-based frontend.

## **7. The interesting ML part: making Supra run on the Pi**

This is where I'd spend the research effort.

Supra's original inference script uses PyTorch, Flan-T5-Base and an SD VAE, with 50 sampling steps recommended. It isn't a drop-in deployment for a 1GB, 32-bit ARM machine.&#x20;

[image](https://www.google.com/s2/favicons?domain=https://huggingface.co\&sz=32)

SupraLabs/Supra2-IMG at main



I'd give Codex the following optimization goals:

| Optimization                      | What it achieves                                     |
| --------------------------------- | ---------------------------------------------------- |
| INT8 / INT4 quantization          | Reduces weight memory and potentially inference cost |
| Export to an ARM-friendly runtime | Avoids relying on full PyTorch on the Pi             |
| Reduce sampling steps             | Fewer transformer evaluations                        |
| Remove unnecessary VAE components | Retain only what image decoding needs                |
| Pruning / low-rank compression    | Reduces model computation if implemented effectively |
| Distillation                      | Potentially make a genuinely smaller, faster model   |

**Quantization by itself won't solve the problem.** Even if the 104M-parameter DiT fits, its repeated processing of image tokens is computationally expensive.

My expectation is that stock Supra inference on the Pi 2 is infeasible in its original form. A compressed implementation might become runnable, but generating an image within 10 minutes is still an ambitious target. We need measured CPU latency and peak memory usage before calling it feasible.

I'd establish the optimization baseline on your RTX 3070 laptop first, then benchmark the same prompts on ARM. A fixed evaluation set would let Codex measure whether faster generation actually preserves recognizable images and prompt adherence.

One further consideration: Supra creates 256 × 256 images, so on your larger HDMI display I'd use a poster-style layout rather than stretching a tiny generated square across the entire screen.

## **8. A practical build sequence**

**Setup checklist**

0/8 complete

Fix the power warning and flash Raspberry Pi OS Lite 32-bit

Connect Pi ↔ laptop over Ethernet and verify SSH

Get a fullscreen image displaying automatically after boot

Create the GitHub project and connect Codex to the Pi

Implement RSS news, prompt creation, and a 10-minute scheduler

Run Supra on the laptop and send generated images to the Pi

Export, compress, and benchmark Supra on ARM

Switch to fully local generation only if performance is sufficient

This sequence matters because the first five steps give you a complete working office display without depending on a difficult model-porting experiment.

A few things I'd add to the final implementation: retain the previous image until the next one finishes, avoid overlapping generation jobs, recover automatically from network failures or power cuts, and include a small *AI-generated illustration* label with the news source and date. An optional Christmas/funny/surreal mode can be selected randomly for each image.

**My choice for your immediate setup:** Pi 2 B + the larger HDMI screen + good 5V/2.5A power supply + Raspberry Pi OS Lite + direct Ethernet to your Ubuntu laptop for development. Let Codex do the software work over SSH, and make Supra optimization the second phase.

That gets you both a useful finished project and a genuinely interesting experiment: *how far can you compress a small diffusion model before a 2015-era Raspberry Pi can generate recognizable images?*
