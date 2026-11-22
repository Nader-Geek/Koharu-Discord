#!/usr/bin/env python3
"""Writes koharu_discord_colab.ipynb, with bot.py embedded, so one notebook
holds the whole Colab setup. Run it after changing bot.py."""

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
BOT = (HERE / "bot.py").read_text(encoding="utf-8")
REPO_SLUG = "Nader-Geek/Koharu-Discord"
REPO = f"https://github.com/{REPO_SLUG}"
BINARY_URL = f"{REPO}/releases/latest/download/koharu-headless"


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip("\n")}


def code(text):
    return {
        "cell_type": "code",
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": text.strip("\n"),
    }


cells = [
    md(
        f"""
# Koharu على Google Colab كـ worker لبوت Discord

النوتبوك ده بيشغّل **koharu-headless** (نسخة بدون نافذة من Koharu مبنية من مكتباته) على GPU الـ Colab، وبيشغّل بوت Discord بيبعت له الصفحات ويرجّع الناتج: ترجمة، تنظيف، OCR، كشف، PSD، CBZ، ووكيل Koharu. الكود والشرح في [{REPO}]({REPO}).

**قبل ما تبدأ**
1. Runtime → Change runtime type → **T4 GPU** (أو أقوى).
2. لازم الـ runtime يكون **Ubuntu 24.04** (الافتراضي الحالي). لو ظهر 22.04 في الخلية الجاية، غيّره من Command Palette → "Change runtime version" لأحدث نسخة.
3. Secrets (أيقونة المفتاح على الشمال): أضف `DISCORD_TOKEN` وفعّل "Notebook access".
4. لازم يكون فيه Release في الريبو فيه ملف `koharu-headless`: ادفع tag زي `v1` (أو Actions → **Build headless** → Run workflow وحمّل الـ artifact). الخلية 3 بتنزّله من آخر Release افتراضيًا. **لو الريبو خاص،** أضف كمان Secret باسم `GITHUB_TOKEN` (Fine-grained token على الريبو بصلاحية Contents: Read) عشان الخلية تقدر تنزّل الملف.

> ⚠️ المسار ده **تجريبي**: تعريف NVIDIA على Colab (580 / CUDA 13.0) أقدم من اللي Koharu بيطلبه (13.3)، والخلية 4 بتتخطى الفحص ده بمتغير بيئة. لو ظهرت أخطاء زي `no kernel image` أو `unsupported PTX`، يبقى التعريف مش كافي وتحتاج جهاز بتعريف R610 أو أحدث.
"""
    ),
    md("## 1) فحص البيئة"),
    code(
        """
import os, subprocess

def sh(command):
    return subprocess.run(command, shell=True, capture_output=True, text=True).stdout

print(sh("nvidia-smi --query-gpu=name,memory.total,driver_version,compute_cap --format=csv"))
print(sh("nvidia-smi | sed -n '3p'"))
print(sh(". /etc/os-release && echo $PRETTY_NAME; ldd --version | head -1; echo CPUs: $(nproc); free -g | sed -n '2p'; df -h /content | tail -1"))
icds = "/usr/share/vulkan/icd.d"
print("Vulkan ICDs:", os.listdir(icds) if os.path.isdir(icds) else "none yet")
"""
    ),
    md(
        """
**المفروض تشوف:** Ubuntu 24.04 و glibc 2.39 (الحزم الأصلية بتاعة Koharu مبنية عليهم)، وتعريف ≥ 580 مع compute capability ≥ 7.5.
- لو التعريف بيقول CUDA 13.3 أو أحدث: احذف `KOHARU_MIN_CUDA_DRIVER` من الخلية 4 (مش هتحتاجه).
- لو 22.04 / glibc 2.35: غيّر نسخة الـ runtime قبل ما تكمل.
"""
    ),
    md("## 2) حزم النظام: Vulkan برمجي للراسم + خطوط عربية و CJK"),
    code(
        """
%%bash
apt-get -qq update
DEBIAN_FRONTEND=noninteractive apt-get -qq install -y --no-install-recommends \\
  libvulkan1 mesa-vulkan-drivers libegl1 libgl1-mesa-dri \\
  fonts-noto-core fonts-noto-cjk > /dev/null
echo "Vulkan ICDs:"; ls /usr/share/vulkan/icd.d
echo "Arabic fonts:"; fc-list | grep -i -E "naskh|noto sans arabic" | head -3
"""
    ),
    md(
        f"""
## 3) ملف `koharu-headless`

الافتراضي هو آخر Release في الريبو. لو بنيته بـ Run workflow بدون tag، حمّل الـ artifact من صفحة الـ run وارفعه إلى Drive، وحط مساره (يبدأ بـ `/`) في `BINARY_URL`. البديل البطيء هو البناء داخل Colab (الخلية اللي بعدها).
"""
    ),
    code(
        f"""
#@title تنزيل الـ binary
BINARY_URL = "{BINARY_URL}"  #@param {{type:"string"}}
import os, shutil, stat, subprocess, requests

BIN = "/content/koharu-headless"
REPO = "{REPO_SLUG}"

def github_token():
    try:
        from google.colab import userdata
        return userdata.get("GITHUB_TOKEN")
    except Exception:
        return None

def download_release_asset(token):
    # A private repo's assets come through the API; requests drops the
    # token on the redirect to GitHub's storage, as it must.
    headers = {{"Authorization": f"Bearer {{token}}", "Accept": "application/vnd.github+json"}}
    release = requests.get(f"https://api.github.com/repos/{{REPO}}/releases/latest", headers=headers, timeout=30)
    release.raise_for_status()
    asset = next(a for a in release.json()["assets"] if a["name"] == "koharu-headless")
    headers["Accept"] = "application/octet-stream"
    with requests.get(asset["url"], headers=headers, stream=True, timeout=600) as response:
        response.raise_for_status()
        with open(BIN, "wb") as out:
            for chunk in response.iter_content(1 << 20):
                out.write(chunk)
    return release.json()["tag_name"]

if BINARY_URL.startswith("/"):
    shutil.copy(BINARY_URL, BIN)
elif BINARY_URL:
    fetched = subprocess.run(["curl", "-fsSL", "--retry", "3", "-o", BIN, BINARY_URL], capture_output=True, text=True)
    if fetched.returncode != 0:
        token = github_token()
        if not token:
            raise SystemExit("التنزيل فشل: لسه مفيش Release، أو الريبو خاص. للريبو الخاص أضف GITHUB_TOKEN في Secrets (Contents: Read على الريبو)، أو حط مسار ملف محلي في BINARY_URL.")
        print("الرابط العام ما اشتغلش؛ بنزّل عبر API بالـ GITHUB_TOKEN...")
        print("release", download_release_asset(token))
elif not os.path.exists(BIN):
    raise SystemExit("حط رابط الـ binary أو مساره في BINARY_URL، أو ابنيه بالخلية التالية")
os.chmod(BIN, os.stat(BIN).st_mode | stat.S_IEXEC)
print(subprocess.run([BIN, "--help"], capture_output=True, text=True).stdout[:1200])
"""
    ),
    code(
        f"""
%%bash
# بديل: البناء داخل Colab (2 vCPU → ساعة أو أكتر، ومحتاج يتعاد كل جلسة إلا لو حفظت الناتج على Drive).
# شيل علامات التعليق:
# curl -sSf https://sh.rustup.rs | sh -s -- -y -q
# apt-get -qq install -y --no-install-recommends clang libclang-dev llvm pkg-config libssl-dev > /dev/null
# git clone --depth 1 {REPO}.git /content/Koharu-Discord
# PATH="$HOME/.cargo/bin:$PATH" bash /content/Koharu-Discord/worker/build.sh -j 2
# cp /content/Koharu-Discord/out/koharu-headless /content/koharu-headless
# # اختياري: mkdir -p /content/drive/MyDrive/koharu && cp /content/koharu-headless /content/drive/MyDrive/koharu/
echo "skipped"
"""
    ),
    md(
        """
## 4) تشغيل الـ worker

- `KOHARU_MIN_CUDA_DRIVER=13000` بيخلي Koharu يقبل تعريف Colab (CUDA 13.0) بدل ما يرفضه.
- المخزن (حزم CUDA/libtorch/llama.cpp والنماذج) بيتحفظ في `/content/koharu-cache` عبر `XDG_CACHE_HOME`. ماتوجّهوش على Google Drive مباشرة (الـ mmap بطيء)؛ لو عايز تحتفظ به بين الجلسات انسخه بـ `rsync` في آخر الجلسة.
- النماذج الافتراضية مناسبة لـ T4 (15 GB): Index-Translate 2B + Hayai OCR + LaMa. جرّب `koharu/index-translate-9b` بعد ما كل حاجة تشتغل.
- **الوكيل (`/ask`) اختياري:** حط نموذج عام بيدعم الأدوات في `AGENT_MODEL` (مثل `koharu/qwen3.5-4b`). بيحتاج ذاكرة إضافية فوق نماذج الترجمة؛ على T4 ابدأ بنموذج 2B-4B أو سيبه فاضي.
"""
    ),
    code(
        """
import os, subprocess, time, requests

WORKER = "http://127.0.0.1:8787"
TRANSLATOR = "koharu/index-translate-2b"  #@param ["koharu/index-translate-2b", "koharu/index-translate-9b", "koharu/gemma4-e4b-it", "koharu/qwen3.5-4b"] {allow-input: true}
AGENT_MODEL = ""  #@param ["", "koharu/qwen3.5-2b", "koharu/qwen3.5-4b", "koharu/gemma4-e4b-it"] {allow-input: true}

os.makedirs("/content/koharu-cache", exist_ok=True)
env = {
    **os.environ,
    "RUST_LOG": "info",
    "XDG_CACHE_HOME": "/content/koharu-cache",
    "KOHARU_MIN_CUDA_DRIVER": "13000",
}
command = [BIN, "--listen", "127.0.0.1:8787", "--jobs", "/content/koharu-jobs",
           "--translator", TRANSLATOR, "--recognizer", "hayai-ocr", "--inpainter", "lama",
           "--fonts", "Noto Naskh Arabic,Noto Sans Arabic,Noto Sans CJK JP"]
if AGENT_MODEL:
    command += ["--agent-model", AGENT_MODEL]
log = open("/content/worker.log", "a", buffering=1)
worker = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT)
for _ in range(60):
    try:
        if requests.get(f"{WORKER}/health", timeout=2).ok:
            break
    except requests.RequestException:
        pass
    time.sleep(1)
else:
    raise SystemExit("الـ worker ما قامش؛ شوف /content/worker.log")
print("worker pid", worker.pid)
print(open("/content/worker.log").read()[-1500:])
"""
    ),
    md("## 5) التسخين: تنزيل الـ runtime (أول مرة ≈ 6 GB، بياخد دقايق)"),
    code(
        """
response = requests.post(f"{WORKER}/warmup", timeout=90 * 60)
print(response.status_code, response.text[:600])
print(open("/content/worker.log").read()[-3000:])
"""
    ),
    md(
        """
لو ظهر `GPU is not available`: التعريف اترفض (شوف الـ log: `CUDA driver is too old`) أو الـ `libcuda.so.1` مش موجودة. لو ظهر `can't be used` مع `couldn't load`: حزمة ما اتحملتش أو glibc أقدم من المطلوب.
"""
    ),
    md("## 6) اختبار سريع بصفحة واحدة قبل Discord"),
    code(
        """
from google.colab import files
from IPython.display import Image, display

uploaded = files.upload()  # ارفع صفحة مانجا واحدة
name, data = next(iter(uploaded.items()))
response = requests.post(f"{WORKER}/jobs", data={"target": "ar"}, files=[("pages", (name, data))])
response.raise_for_status()
job = response.json()["id"]
while True:
    status = requests.get(f"{WORKER}/jobs/{job}").json()["status"]
    print(status["state"], status.get("finished"), "/", status.get("steps"), status.get("current") or "")
    if status["state"] in ("done", "failed", "cancelled"):
        break
    time.sleep(10)
if status["state"] == "done":
    for suffix, file in (("", "page1.png"), ("/cleaned", "page1-cleaned.png")):
        open(f"/content/{file}", "wb").write(requests.get(f"{WORKER}/jobs/{job}/pages/1{suffix}").content)
        display(Image(f"/content/{file}", width=420))
    for block in status["pages"][0]["blocks"]:
        print(block["role"], block["source"], "→", block["translation"])
else:
    print(status["state"], status.get("error"))
    print(open("/content/worker.log").read()[-4000:])
"""
    ),
    md(
        """
## 7) بوت Discord

**في Discord Developer Portal:** New Application → Bot → انسخ الـ Token إلى Secrets باسم `DISCORD_TOKEN`. مش محتاج Message Content Intent (الأوامر slash و context menu). رابط الدعوة من OAuth2 → URL Generator بالـ scopes `bot` + `applications.commands` وصلاحيات Send Messages وAttach Files وRead Message History.

**الأوامر:**

| الأمر | يعمل |
|---|---|
| `/translate` | كشف + OCR + تنظيف + ترجمة (خيارات: اللغة، النموذج، `cleanup`، تعليمات، شكل الناتج PNG/CBZ/PSD/نص) |
| `/clean` | تنظيف الصفحات بدون ترجمة (LaMa أو نموذج توليدي) |
| `/ocr` | قراءة النص الأصلي بترتيب القراءة، مع صورة مرقّمة اختياريًا |
| `/detect` | صورة بالصناديق المرقّمة: نصوص (أزرق)، SFX (برتقالي)، فقاعات (أخضر)، لوحات (رمادي) |
| كليك يمين على رسالة → Apps | **Translate with Koharu** / **Clean with Koharu** |
| `/redo` | إعادة مراحل على مهمة منتهية: ترجمة فقط بتعليمات أو لغة أو نموذج جديد، أو OCR+ترجمة، أو تنظيف فقط، أو الكل |
| `/export` | ناتج مهمة منتهية بشكل آخر: الصفحات، الصفحات المنظّفة فقط، CBZ، PSD، النصوص، صورة الكشف |
| `/ask` | وكيل Koharu على مشروع المهمة (يتطلب `AGENT_MODEL`) |
| `/status` `/cancel` `/models` `/help` | متابعة |
| `/settings` `/glossary` | اللغة والنموذج والتعليمات الافتراضية للسيرفر، وقاموس أسماء يُرسل مع كل ترجمة |
| `/worker` | حالة الـ worker، warmup، unload، وتغيير النماذج الافتراضية (لمن لديه Manage Server) |

الخلية دي بتفضل شغالة طول ما البوت شغال؛ لإيقافه اعمل Interrupt.
"""
    ),
    code("%%writefile /content/bot.py\n" + BOT),
    code(
        """
%pip -q install -U "discord.py>=2.4" aiohttp pillow
import importlib, os, sys
from google.colab import userdata

os.environ["DISCORD_TOKEN"] = userdata.get("DISCORD_TOKEN")
os.environ["KOHARU_WORKER"] = WORKER
os.environ["KOHARU_TARGET"] = "ar"
os.environ["KOHARU_SETTINGS"] = "/content/koharu_settings.json"
sys.path.insert(0, "/content")
import bot as koharu_bot
importlib.reload(koharu_bot)
await koharu_bot.bot.start(os.environ["DISCORD_TOKEN"])
"""
    ),
    md("## 8) صيانة"),
    code(
        """
# آخر سطور الـ log
print(open("/content/worker.log").read()[-4000:])
"""
    ),
    code(
        """
# المهام الحالية على الـ worker، وتفريغ النماذج من ذاكرة الـ GPU
print(requests.get(f"{WORKER}/jobs").json())
print(requests.post(f"{WORKER}/unload").text)
"""
    ),
    code(
        """
# اختياري: حفظ المخزن على Drive عشان الجلسة الجاية ما تنزّلش كل حاجة تاني
# from google.colab import drive; drive.mount("/content/drive")
# !mkdir -p /content/drive/MyDrive/koharu-cache && rsync -a --info=progress2 /content/koharu-cache/ /content/drive/MyDrive/koharu-cache/
# وفي بداية الجلسة الجاية، قبل الخلية 4:
# !rsync -a /content/drive/MyDrive/koharu-cache/ /content/koharu-cache/
"""
    ),
    code(
        """
# إيقاف الـ worker
worker.terminate(); print(worker.wait(timeout=30))
"""
    ),
]

notebook = {
    "cells": cells,
    "metadata": {
        "accelerator": "GPU",
        "colab": {"gpuType": "T4", "provenance": []},
        "kernelspec": {"display_name": "Python 3", "name": "python3"},
        "language_info": {"name": "python"},
    },
    "nbformat": 4,
    "nbformat_minor": 0,
}

out = HERE / "koharu_discord_colab.ipynb"
out.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"wrote {out} with {len(cells)} cells")
