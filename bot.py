"""Discord front end for koharu-headless.

Every image, ZIP, or CBZ a user hands the bot becomes a job on the worker:
a translation, a clean-up, an OCR pass, or a detection, whose pages come back
in the channel as PNGs, a CBZ, layered PSDs, or text. A finished job stays
open on the worker, so it can be run again, exported in another format, or
handed to Koharu's agent.

Environment:
  DISCORD_TOKEN     the bot's token (Discord Developer Portal → Bot)
  KOHARU_WORKER     the worker's URL, http://127.0.0.1:8787 by default
  KOHARU_TARGET     the default target language, `ar` by default
  KOHARU_SETTINGS   where each server's defaults and glossary are kept

Run it beside the worker: `python bot.py`. In a notebook, whose event loop is
already running, import it and `await bot.start(token)` instead.
"""

import asyncio
import io
import json
import os
import time
from pathlib import Path
from typing import Optional

import aiohttp
import discord
from discord import app_commands
from discord.ext import commands
from PIL import Image, ImageDraw, ImageFont

WORKER = os.environ.get("KOHARU_WORKER", "http://127.0.0.1:8787").rstrip("/")
DEFAULT_TARGET = os.environ.get("KOHARU_TARGET", "ar")
SETTINGS_PATH = Path(os.environ.get("KOHARU_SETTINGS", "koharu_settings.json"))

POLL_SECONDS = 5
EDIT_SECONDS = 10
# Discord takes 10 attachments a message, and 10 MB a file from a bot on an
# unboosted server.
FILES_PER_MESSAGE = 8
FILE_BYTES = 9_500_000
PAGE_TYPES = (".png", ".jpg", ".jpeg", ".webp", ".zip", ".cbz")

# What each command runs, as the pipeline names its stages.
STAGES = {
    "translate": "detect,recognize,inpaint,translate",
    "translate_keep": "detect,recognize,translate",
    "clean": "detect,inpaint",
    "ocr": "detect,recognize",
    "detect": "detect",
}
KIND_LABELS = {"translate": "ترجمة", "clean": "تنظيف", "ocr": "OCR", "detect": "كشف", "redo": "إعادة"}

LANGUAGES = [
    ("العربية (ar)", "ar"), ("English (en)", "en"), ("日本語 (ja)", "ja"),
    ("简体中文 (zh-Hans)", "zh-Hans"), ("繁體中文 (zh-Hant)", "zh-Hant"), ("한국어 (ko)", "ko"),
    ("Français (fr)", "fr"), ("Deutsch (de)", "de"), ("Español (es)", "es"),
    ("Português (pt-BR)", "pt-BR"), ("Italiano (it)", "it"), ("Русский (ru)", "ru"),
    ("Türkçe (tr)", "tr"), ("Bahasa Indonesia (id)", "id"), ("Tiếng Việt (vi)", "vi"),
    ("ไทย (th)", "th"), ("Polski (pl)", "pl"), ("Nederlands (nl)", "nl"), ("فارسی (fa)", "fa"),
    ("हिन्दी (hi)", "hi"), ("Українська (uk)", "uk"), ("Filipino (fil)", "fil"),
    ("Bahasa Melayu (ms)", "ms"), ("Ελληνικά (el)", "el"), ("Svenska (sv)", "sv"),
]
INPAINTERS = [("LaMa (سريع، افتراضي)", "lama"), ("FLUX.3 Klein (توليدي)", "flux3-klein"), ("Qwen-Image 2.1 Turbo (توليدي)", "qwen-image")]
RECOGNIZERS = [("Hayai OCR (ياباني/صيني/كوري/إنجليزي)", "hayai-ocr"), ("Manga OCR AR (ياباني)", "manga-ocr-ar"), ("Baberu OCR (ياباني)", "baberu-ocr")]
REASONING = [("إيقاف", "off"), ("منخفض", "low"), ("متوسط", "medium"), ("عالٍ", "high")]


class Settings:
    """Each server's defaults and glossary, in one JSON file."""

    def __init__(self, path: Path):
        self.path = path
        self.data = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}

    @staticmethod
    def scope(interaction: discord.Interaction) -> str:
        return str(interaction.guild_id) if interaction.guild_id else f"dm:{interaction.user.id}"

    def get(self, interaction: discord.Interaction) -> dict:
        return {"target": DEFAULT_TARGET, "model": None, "instructions": "", "glossary": {},
                **self.data.get(self.scope(interaction), {})}

    def update(self, interaction: discord.Interaction, **changes) -> dict:
        entry = self.get(interaction)
        entry.update(changes)
        self.data[self.scope(interaction)] = entry
        self.path.write_text(json.dumps(self.data, ensure_ascii=False, indent=1), encoding="utf-8")
        return entry


settings = Settings(SETTINGS_PATH)


def instructions_for(entry: dict, extra: str = "") -> str:
    """The server's instructions and glossary, then the request's own."""
    parts = []
    if entry.get("instructions"):
        parts.append(entry["instructions"])
    if entry.get("glossary"):
        lines = "\n".join(f"- {term} → {rendering}" for term, rendering in entry["glossary"].items())
        parts.append("Glossary; keep these renderings exactly:\n" + lines)
    if extra and extra.strip():
        parts.append(extra.strip())
    return "\n\n".join(parts)


class KoharuBot(commands.Bot):
    def __init__(self):
        super().__init__(command_prefix="!", intents=discord.Intents.default())
        self.http_client: Optional[aiohttp.ClientSession] = None
        self._models: tuple[float, Optional[dict]] = (0.0, None)

    async def setup_hook(self):
        # Long reads: a page can take minutes on a small GPU, and an ask longer.
        timeout = aiohttp.ClientTimeout(total=None, sock_connect=30, sock_read=3600)
        self.http_client = aiohttp.ClientSession(timeout=timeout)
        self.tree.add_command(SettingsGroup())
        self.tree.add_command(GlossaryGroup())
        self.tree.add_command(WorkerGroup())
        synced = await self.tree.sync()
        print(f"{len(synced)} commands synced")

    async def close(self):
        if self.http_client:
            await self.http_client.close()
        await super().close()

    async def models(self) -> Optional[dict]:
        """The worker's model list, kept a minute."""
        at, cached = self._models
        if cached is not None and time.monotonic() - at < 60:
            return cached
        try:
            async with self.http_client.get(f"{WORKER}/models") as response:
                if response.status == 200:
                    cached = await response.json()
                    self._models = (time.monotonic(), cached)
                    return cached
        except aiohttp.ClientError:
            pass
        return cached


bot = KoharuBot()


@bot.event
async def on_ready():
    print(f"logged in as {bot.user} · worker at {WORKER}")


# --- Autocomplete and choices ---

async def language_autocomplete(interaction: discord.Interaction, current: str):
    current = current.lower()
    found = [(name, code) for name, code in LANGUAGES if current in name.lower() or current in code.lower()]
    if current and not found:
        found = [(current, current)]
    return [app_commands.Choice(name=name, value=code) for name, code in found[:25]]


async def model_autocomplete(interaction: discord.Interaction, current: str):
    listed = await bot.models()
    models = (listed or {}).get("translators", [])
    current = current.lower()
    found = [m for m in models if current in m["reference"].lower() or current in m["name"].lower()]
    choices = [app_commands.Choice(name=f"{m['name']} · {m['reference']}"[:100], value=m["reference"]) for m in found[:25]]
    if current and not choices:
        choices = [app_commands.Choice(name=current, value=current)]
    return choices


def choices(pairs):
    return [app_commands.Choice(name=name, value=value) for name, value in pairs]


OUTPUTS = [("صور الصفحات", "pages"), ("CBZ", "cbz"), ("PSD بطبقات", "psd"), ("نص فقط", "text")]
EXPORTS = [("الصفحات النهائية", "pages"), ("الصفحات بعد التنظيف فقط", "cleaned"), ("CBZ", "cbz"), ("PSD بطبقات", "psd"), ("النصوص", "text"), ("صورة الكشف بالأرقام", "overlay")]


# --- Commands ---

PAGE_PARAMS = dict(page1="صفحة (PNG/JPG/WebP)", page2="صفحة", page3="صفحة", page4="صفحة", page5="صفحة",
                   archive="أرشيف ZIP أو CBZ لفصل كامل")


@bot.tree.command(name="translate", description="ترجم صفحات مانجا: كشف + OCR + تنظيف + ترجمة")
@app_commands.describe(**PAGE_PARAMS, target="اللغة الهدف (BCP 47)", model="نموذج الترجمة",
                       cleanup="إزالة النص الأصلي قبل كتابة الترجمة (افتراضي نعم)",
                       instructions="تعليمات إضافية للمترجم: أسماء، نبرة، مصطلحات", output="شكل الناتج")
@app_commands.autocomplete(target=language_autocomplete, model=model_autocomplete)
@app_commands.choices(output=choices(OUTPUTS))
async def translate(interaction: discord.Interaction,
                    page1: Optional[discord.Attachment] = None, page2: Optional[discord.Attachment] = None,
                    page3: Optional[discord.Attachment] = None, page4: Optional[discord.Attachment] = None,
                    page5: Optional[discord.Attachment] = None, archive: Optional[discord.Attachment] = None,
                    target: Optional[str] = None, model: Optional[str] = None, cleanup: bool = True,
                    instructions: str = "", output: str = "pages"):
    entry = settings.get(interaction)
    await submit(interaction, attached(page1, page2, page3, page4, page5, archive), "translate",
                 stages=STAGES["translate" if cleanup else "translate_keep"],
                 target=target or entry["target"], instructions=instructions_for(entry, instructions),
                 models={"translator": model or entry["model"]}, output=output)


@bot.tree.command(name="clean", description="نظّف الصفحات: كشف النصوص وإزالتها وإعادة رسم ما خلفها، بدون ترجمة")
@app_commands.describe(**PAGE_PARAMS, model="نموذج الترميم", output="شكل الناتج")
@app_commands.choices(model=choices(INPAINTERS), output=choices([o for o in OUTPUTS if o[1] != "text"]))
async def clean(interaction: discord.Interaction,
                page1: Optional[discord.Attachment] = None, page2: Optional[discord.Attachment] = None,
                page3: Optional[discord.Attachment] = None, page4: Optional[discord.Attachment] = None,
                page5: Optional[discord.Attachment] = None, archive: Optional[discord.Attachment] = None,
                model: Optional[str] = None, output: str = "pages"):
    await submit(interaction, attached(page1, page2, page3, page4, page5, archive), "clean",
                 stages=STAGES["clean"], target=DEFAULT_TARGET, instructions="",
                 models={"inpainter": model}, output=output)


@bot.tree.command(name="ocr", description="اقرأ النص الأصلي من الصفحات بترتيب القراءة")
@app_commands.describe(**PAGE_PARAMS, model="نموذج OCR", overlay="إرفاق صورة بالصناديق مرقّمة")
@app_commands.choices(model=choices(RECOGNIZERS))
async def ocr(interaction: discord.Interaction,
              page1: Optional[discord.Attachment] = None, page2: Optional[discord.Attachment] = None,
              page3: Optional[discord.Attachment] = None, page4: Optional[discord.Attachment] = None,
              page5: Optional[discord.Attachment] = None, archive: Optional[discord.Attachment] = None,
              model: Optional[str] = None, overlay: bool = False):
    await submit(interaction, attached(page1, page2, page3, page4, page5, archive), "ocr",
                 stages=STAGES["ocr"], target=DEFAULT_TARGET, instructions="",
                 models={"recognizer": model}, output="text", overlay=overlay)


@bot.tree.command(name="detect", description="اكشف اللوحات والفقاعات والنصوص وأرسل صورة بالصناديق مرقّمة")
@app_commands.describe(**PAGE_PARAMS)
async def detect(interaction: discord.Interaction,
                 page1: Optional[discord.Attachment] = None, page2: Optional[discord.Attachment] = None,
                 page3: Optional[discord.Attachment] = None, page4: Optional[discord.Attachment] = None,
                 page5: Optional[discord.Attachment] = None, archive: Optional[discord.Attachment] = None):
    await submit(interaction, attached(page1, page2, page3, page4, page5, archive), "detect",
                 stages=STAGES["detect"], target=DEFAULT_TARGET, instructions="", models={}, output="overlay")


@bot.tree.context_menu(name="Translate with Koharu")
async def translate_message(interaction: discord.Interaction, message: discord.Message):
    """Right-click a message holding page images or an archive → Apps."""
    entry = settings.get(interaction)
    await submit(interaction, pages_in(message), "translate", stages=STAGES["translate"],
                 target=entry["target"], instructions=instructions_for(entry),
                 models={"translator": entry["model"]}, output="pages")


@bot.tree.context_menu(name="Clean with Koharu")
async def clean_message(interaction: discord.Interaction, message: discord.Message):
    await submit(interaction, pages_in(message), "clean", stages=STAGES["clean"],
                 target=DEFAULT_TARGET, instructions="", models={}, output="pages")


REDO_ACTIONS = [("ترجمة فقط (بإعدادات جديدة)", "translate"), ("OCR + ترجمة", "recognize,translate"),
                ("تنظيف فقط", "inpaint"), ("كل المراحل من البداية", "detect,recognize,inpaint,translate")]


@bot.tree.command(name="redo", description="أعد تشغيل مراحل على مهمة منتهية بنفس الصفحات، مثلًا الترجمة بتعليمات جديدة")
@app_commands.describe(job="رقم المهمة", action="ما يُعاد", target="اللغة الهدف", model="نموذج الترجمة",
                       inpainter="نموذج الترميم", instructions="تعليمات إضافية", output="شكل الناتج")
@app_commands.autocomplete(target=language_autocomplete, model=model_autocomplete)
@app_commands.choices(action=choices(REDO_ACTIONS), inpainter=choices(INPAINTERS), output=choices(OUTPUTS))
async def redo(interaction: discord.Interaction, job: int, action: str = "translate",
               target: Optional[str] = None, model: Optional[str] = None, inpainter: Optional[str] = None,
               instructions: Optional[str] = None, output: str = "pages"):
    entry = settings.get(interaction)
    body = {"stages": action}
    if target:
        body["target"] = target
    if model:
        body["translator"] = model
    if inpainter:
        body["inpainter"] = inpainter
    if instructions is not None:
        body["instructions"] = instructions_for(entry, instructions)
    await interaction.response.defer(thinking=True)
    status, text = await call("post", f"/jobs/{job}/rerun", json=body)
    if status != 200:
        await interaction.followup.send(f"الـ worker رفض الطلب ({status}): {text[:1500]}")
        return
    kind = "clean" if action == "inpaint" else "translate"
    message = await interaction.channel.send(f"🧾 المهمة #{job} · إعادة ({action}) · في الانتظار...")
    await interaction.followup.send(f"أعدت تشغيل المهمة #{job}. التقدم هنا: {message.jump_url}", ephemeral=True)
    info = await follow(job, message, "redo")
    if info:
        await deliver(interaction.channel, job, info, kind, output)


@bot.tree.command(name="export", description="أرسل ناتج مهمة منتهية بشكل آخر بدون إعادة معالجة")
@app_commands.describe(job="رقم المهمة", format="الشكل المطلوب")
@app_commands.choices(format=choices(EXPORTS))
async def export(interaction: discord.Interaction, job: int, format: str = "pages"):
    await interaction.response.defer(thinking=True)
    status, text = await call("get", f"/jobs/{job}")
    if status != 200:
        await interaction.followup.send(f"({status}) {text[:1500]}")
        return
    info = json.loads(text)
    if info["status"]["state"] != "done":
        await interaction.followup.send(f"المهمة #{job} حالتها `{info['status']['state']}`؛ التصدير للمهام المنتهية فقط.")
        return
    await interaction.followup.send(f"بجهّز {format} للمهمة #{job}...", ephemeral=True)
    translated = any(b.get("translation") for p in info["status"]["pages"] for b in p["blocks"])
    if format == "overlay":
        kind = "detect"
    elif format == "text":
        kind = "translate" if translated else "ocr"
    else:
        # The files alone; the texts came with the job, or come with `text`.
        kind = "export"
    await deliver(interaction.channel, job, info, kind, format)


@bot.tree.command(name="status", description="حالة مهمة، أو آخر المهام")
@app_commands.describe(job="رقم المهمة (اختياري)")
async def status_command(interaction: discord.Interaction, job: Optional[int] = None):
    await interaction.response.defer(thinking=True, ephemeral=True)
    if job is None:
        status, text = await call("get", "/jobs")
        if status != 200:
            await interaction.followup.send(f"({status}) {text[:1500]}")
            return
        jobs = json.loads(text)[:15]
        if not jobs:
            await interaction.followup.send("مفيش مهام لسه.")
            return
        lines = [f"#{j['id']} · {j['state']} · {j['pages']} صفحة · {','.join(j['stages'])} → `{j['target']}`" for j in jobs]
        await interaction.followup.send("\n".join(lines))
        return
    status, text = await call("get", f"/jobs/{job}")
    if status != 200:
        await interaction.followup.send(f"({status}) {text[:1500]}")
        return
    info = json.loads(text)
    state = info["status"]
    detail = describe(state, info["pages"], info["target"])
    if state["state"] == "failed":
        detail += f"\n```{state['error'][:1200]}```"
    await interaction.followup.send(f"المهمة #{job} · {','.join(info['stages'])} · {detail}")


@bot.tree.command(name="cancel", description="أوقف مهمة في الانتظار أو قيد التشغيل")
@app_commands.describe(job="رقم المهمة")
async def cancel(interaction: discord.Interaction, job: int):
    status, text = await call("post", f"/jobs/{job}/cancel")
    await interaction.response.send_message(
        f"بيتم إيقاف المهمة #{job}." if status == 200 else f"({status}) {text[:1500]}", ephemeral=True)


@bot.tree.command(name="ask", description="اطلب من وكيل Koharu فحص أو تعديل مشروع مهمة منتهية (يتطلب نموذج وكيل على الـ worker)")
@app_commands.describe(job="رقم المهمة", prompt="الطلب، مثلًا: راجع ترجمة صفحة 2 وصحّح أسماء الشخصيات", reasoning="مستوى التفكير")
@app_commands.choices(reasoning=choices(REASONING))
async def ask(interaction: discord.Interaction, job: int, prompt: str, reasoning: Optional[str] = None):
    await interaction.response.defer(thinking=True)
    message = await interaction.channel.send(f"🤖 الوكيل بيشتغل على المهمة #{job}...")
    await interaction.followup.send(f"بعت الطلب للوكيل. الرد هنا: {message.jump_url}", ephemeral=True)
    body = {"prompt": prompt}
    if reasoning:
        body["reasoning"] = reasoning
    status, text = await call("post", f"/jobs/{job}/ask", json=body)
    if status != 200:
        await message.edit(content=f"❌ الوكيل فشل ({status}):\n```{text[:1500]}```")
        return
    reply = json.loads(text)
    calls = reply.get("calls", [])
    used = ", ".join(sorted({c.get("tool", "?") for c in calls if c.get("type") == "tool_started"})) or "لا شيء"
    answer = reply.get("answer", "").strip() or "(بدون رد نصي)"
    await message.edit(content=f"🤖 **المهمة #{job}** · الأدوات المستخدمة: {used}\n\n{answer[:1800]}")
    if any(c.get("tool") == "edit" for c in calls):
        await interaction.channel.send(f"الوكيل عدّل المشروع؛ الصفحات اتعاد رسمها. اعرضها بـ `/export job:{job}`.")


@bot.tree.command(name="models", description="النماذج المتاحة على الـ worker والمستخدمة حاليًا")
async def models_command(interaction: discord.Interaction):
    await interaction.response.defer(thinking=True, ephemeral=True)
    listed = await bot.models()
    if not listed:
        await interaction.followup.send(f"مش قادر أوصل للـ worker على {WORKER}.")
        return
    current = listed["current"]
    translators = "\n".join(f"- `{m['reference']}` {m['name']}" + (" 🛠" if m.get("tools") else "") for m in listed["translators"][:40])
    text = (f"**المستخدم الآن:** ترجمة `{current['translator']}` · OCR `{current['recognizer']}` · ترميم `{current['inpainter']}`"
            f" · وكيل `{listed.get('agent_model') or 'غير مفعّل'}`\n"
            f"**نماذج الترجمة** (🛠 = تدعم الأدوات/الوكيل):\n{translators}\n"
            f"**OCR:** {', '.join('`' + m['id'] + '`' for m in listed['recognizers'])}\n"
            f"**ترميم:** {', '.join('`' + m['id'] + '`' for m in listed['inpainters'])}")
    await interaction.followup.send(text[:1990])


@bot.tree.command(name="help", description="شرح أوامر Koharu")
async def help_command(interaction: discord.Interaction):
    text = (
        "**المعالجة** (ارفع حتى 5 صور أو ملف ZIP/CBZ):\n"
        "`/translate` كشف + OCR + تنظيف + ترجمة · `/clean` تنظيف بدون ترجمة · `/ocr` قراءة النص فقط · `/detect` صورة بالصناديق\n"
        "أو كليك يمين على رسالة فيها صور → Apps → **Translate / Clean with Koharu**\n"
        "**على مهمة منتهية:** `/redo` إعادة مراحل (مثلًا ترجمة بتعليمات جديدة) · `/export` ناتج بشكل آخر (PNG/CBZ/PSD/نص/صفحات منظفة) · `/ask` وكيل Koharu\n"
        "**متابعة:** `/status` · `/cancel` · `/models`\n"
        "**إعدادات السيرفر:** `/settings` (اللغة والنموذج والتعليمات الافتراضية) · `/glossary` (قاموس أسماء ومصطلحات يُرسل مع كل ترجمة)\n"
        "**إدارة الـ worker:** `/worker status|warmup|unload|translator|recognizer|inpainter`"
    )
    await interaction.response.send_message(text, ephemeral=True)


@app_commands.default_permissions(manage_guild=True)
class SettingsGroup(app_commands.Group):
    """Each server's defaults."""

    def __init__(self):
        super().__init__(name="settings", description="إعدادات Koharu الافتراضية في هذا السيرفر")

    @app_commands.command(name="show", description="اعرض الإعدادات الحالية")
    async def show(self, interaction: discord.Interaction):
        entry = settings.get(interaction)
        glossary = len(entry["glossary"])
        await interaction.response.send_message(
            f"اللغة الهدف: `{entry['target']}` · نموذج الترجمة: `{entry['model'] or 'افتراضي الـ worker'}` · "
            f"القاموس: {glossary} مدخل\nالتعليمات:\n{entry['instructions'] or '(لا شيء)'}", ephemeral=True)

    @app_commands.command(name="target", description="اللغة الهدف الافتراضية")
    @app_commands.autocomplete(target=language_autocomplete)
    async def target(self, interaction: discord.Interaction, target: str):
        settings.update(interaction, target=target)
        await interaction.response.send_message(f"اللغة الهدف الافتراضية صارت `{target}`.", ephemeral=True)

    @app_commands.command(name="model", description="نموذج الترجمة الافتراضي لهذا السيرفر")
    @app_commands.autocomplete(model=model_autocomplete)
    async def model(self, interaction: discord.Interaction, model: str):
        settings.update(interaction, model=model)
        await interaction.response.send_message(f"نموذج الترجمة الافتراضي صار `{model}`.", ephemeral=True)

    @app_commands.command(name="instructions", description="تعليمات تُرسل مع كل ترجمة (النبرة، الأسلوب، ما لا يُترجم)")
    async def instructions(self, interaction: discord.Interaction, text: str):
        settings.update(interaction, instructions=text.strip())
        await interaction.response.send_message("اتحفظت التعليمات.", ephemeral=True)

    @app_commands.command(name="reset", description="امسح إعدادات السيرفر والقاموس")
    async def reset(self, interaction: discord.Interaction):
        settings.update(interaction, target=DEFAULT_TARGET, model=None, instructions="", glossary={})
        await interaction.response.send_message("اترجّعت الإعدادات للافتراضي.", ephemeral=True)


class GlossaryGroup(app_commands.Group):
    """Names and terms every translation keeps."""

    def __init__(self):
        super().__init__(name="glossary", description="قاموس أسماء ومصطلحات يُرسل مع كل ترجمة في هذا السيرفر")

    @app_commands.command(name="add", description="أضف مصطلحًا وترجمته")
    async def add(self, interaction: discord.Interaction, term: str, translation: str):
        entry = settings.get(interaction)
        if len(entry["glossary"]) >= 200 and term not in entry["glossary"]:
            await interaction.response.send_message("القاموس وصل 200 مدخل.", ephemeral=True)
            return
        entry["glossary"][term.strip()] = translation.strip()
        settings.update(interaction, glossary=entry["glossary"])
        await interaction.response.send_message(f"أضفت: {term} → {translation}", ephemeral=True)

    @app_commands.command(name="remove", description="احذف مصطلحًا")
    async def remove(self, interaction: discord.Interaction, term: str):
        entry = settings.get(interaction)
        removed = entry["glossary"].pop(term.strip(), None)
        settings.update(interaction, glossary=entry["glossary"])
        await interaction.response.send_message("اتحذف." if removed is not None else "مش موجود.", ephemeral=True)

    @app_commands.command(name="list", description="اعرض القاموس")
    async def list_(self, interaction: discord.Interaction):
        entry = settings.get(interaction)
        if not entry["glossary"]:
            await interaction.response.send_message("القاموس فاضي. أضف بـ `/glossary add`.", ephemeral=True)
            return
        lines = "\n".join(f"- {t} → {r}" for t, r in entry["glossary"].items())
        await interaction.response.send_message(lines[:1990], ephemeral=True)

    @app_commands.command(name="clear", description="امسح القاموس (Manage Server)")
    async def clear(self, interaction: discord.Interaction):
        # Discord lets only a top-level command carry default permissions, so
        # a subcommand checks itself.
        member = interaction.user
        if interaction.guild and not getattr(member, "guild_permissions", None).manage_guild:
            await interaction.response.send_message("محتاج صلاحية Manage Server.", ephemeral=True)
            return
        settings.update(interaction, glossary={})
        await interaction.response.send_message("اتمسح القاموس.", ephemeral=True)


@app_commands.default_permissions(manage_guild=True)
class WorkerGroup(app_commands.Group):
    """The worker itself."""

    def __init__(self):
        super().__init__(name="worker", description="إدارة الـ worker: الحالة والنماذج والذاكرة")

    @app_commands.command(name="status", description="هل الـ worker شغال، وما النماذج المستخدمة")
    async def status(self, interaction: discord.Interaction):
        await interaction.response.defer(thinking=True, ephemeral=True)
        status, text = await call("get", "/health")
        if status != 200:
            await interaction.followup.send(f"الـ worker مش بيرد على {WORKER} ({status}): {text[:500]}")
            return
        listed = await bot.models()
        current = (listed or {}).get("current", {})
        jobs_status, jobs_text = await call("get", "/jobs")
        active = [j for j in (json.loads(jobs_text) if jobs_status == 200 else []) if j["state"] in ("queued", "running", "rendering")]
        await interaction.followup.send(
            f"✅ شغال على {WORKER}\nترجمة `{current.get('translator')}` · OCR `{current.get('recognizer')}` · ترميم `{current.get('inpainter')}`\n"
            f"مهام نشطة: {len(active)}")

    @app_commands.command(name="warmup", description="نزّل وحمّل الـ runtime على الـ GPU الآن بدل أول مهمة")
    async def warmup(self, interaction: discord.Interaction):
        await interaction.response.defer(thinking=True, ephemeral=True)
        status, text = await call("post", "/warmup")
        await interaction.followup.send(f"({status}) {text[:1500]}")

    @app_commands.command(name="unload", description="فرّغ كل النماذج من ذاكرة الـ GPU (بتتحمل تاني عند الحاجة)")
    async def unload(self, interaction: discord.Interaction):
        status, text = await call("post", "/unload")
        await interaction.response.send_message(f"({status}) {text[:500]}", ephemeral=True)

    @app_commands.command(name="translator", description="غيّر نموذج الترجمة الافتراضي للـ worker")
    @app_commands.autocomplete(model=model_autocomplete)
    async def translator(self, interaction: discord.Interaction, model: str):
        await self._config(interaction, {"translator": model})

    @app_commands.command(name="recognizer", description="غيّر نموذج OCR الافتراضي للـ worker")
    @app_commands.choices(model=choices(RECOGNIZERS))
    async def recognizer(self, interaction: discord.Interaction, model: str):
        await self._config(interaction, {"recognizer": model})

    @app_commands.command(name="inpainter", description="غيّر نموذج الترميم الافتراضي للـ worker")
    @app_commands.choices(model=choices(INPAINTERS))
    async def inpainter(self, interaction: discord.Interaction, model: str):
        await self._config(interaction, {"inpainter": model})

    async def _config(self, interaction: discord.Interaction, body: dict):
        status, text = await call("post", "/config", json=body)
        bot._models = (0.0, None)
        await interaction.response.send_message(
            f"الإعدادات الحالية: `{text[:800]}`" if status == 200 else f"({status}) {text[:800]}", ephemeral=True)


# --- Jobs ---

def attached(*attachments: Optional[discord.Attachment]) -> list:
    return [a for a in attachments if a is not None]


def pages_in(message: discord.Message) -> list:
    return [a for a in message.attachments if a.filename.lower().endswith(PAGE_TYPES)]


async def call(method: str, path: str, **kwargs) -> tuple[int, str]:
    """One request to the worker: its status and body."""
    try:
        async with bot.http_client.request(method, f"{WORKER}{path}", **kwargs) as response:
            return response.status, await response.text()
    except aiohttp.ClientError as error:
        return 0, f"مش قادر أوصل للـ worker على {WORKER}: {error}"


async def fetch(path: str) -> Optional[bytes]:
    try:
        async with bot.http_client.get(f"{WORKER}{path}") as response:
            if response.status == 200:
                return await response.read()
    except aiohttp.ClientError:
        pass
    return None


async def submit(interaction: discord.Interaction, attachments: list, kind: str, *, stages: str, target: str,
                 instructions: str, models: dict, output: str, overlay: bool = False):
    """Uploads the pages as a job, follows it in the channel, and delivers its output."""
    if not attachments:
        await interaction.response.send_message("ابعت صورة واحدة على الأقل (PNG/JPG/WebP) أو ملف ZIP/CBZ.", ephemeral=True)
        return
    await interaction.response.defer(thinking=True)
    form = aiohttp.FormData()
    form.add_field("target", target)
    form.add_field("stages", stages)
    form.add_field("instructions", instructions or "")
    for name, model in models.items():
        if model:
            form.add_field(name, model)
    for attachment in attachments:
        form.add_field("pages", await attachment.read(), filename=attachment.filename,
                       content_type=attachment.content_type or "application/octet-stream")
    status, text = await call("post", "/jobs", data=form)
    if status != 200:
        await interaction.followup.send(f"الـ worker رفض الطلب ({status}): {text[:1500]}")
        return
    job = json.loads(text)
    job_id, pages = job["id"], job["pages"]
    # A channel message outlives the interaction's 15-minute token, so
    # progress and results go there; the followup only says where.
    message = await interaction.channel.send(f"🧾 المهمة #{job_id} · {KIND_LABELS[kind]} · {pages} صفحة · في الانتظار...")
    await interaction.followup.send(f"بدأت المهمة #{job_id} ({KIND_LABELS[kind]}). التقدم هنا: {message.jump_url}", ephemeral=True)
    info = await follow(job_id, message, kind)
    if info:
        await deliver(interaction.channel, job_id, info, kind, output, overlay)


async def follow(job_id: int, message: discord.Message, kind: str) -> Optional[dict]:
    """Edits `message` with the job's progress until it ends; the job's info once done."""
    last_edit = 0.0
    while True:
        await asyncio.sleep(POLL_SECONDS)
        status, text = await call("get", f"/jobs/{job_id}")
        if status != 200:
            await message.edit(content=f"❌ المهمة #{job_id}: {text[:1000]}")
            return None
        info = json.loads(text)
        state = info["status"]
        if state["state"] in ("done", "failed", "cancelled"):
            break
        if time.monotonic() - last_edit > EDIT_SECONDS:
            await message.edit(content=f"🧾 المهمة #{job_id} · {KIND_LABELS[kind]} · {describe(state, info['pages'], info['target'])}")
            last_edit = time.monotonic()
    if state["state"] == "failed":
        await message.edit(content=f"❌ المهمة #{job_id} فشلت:\n```{state['error'][:1500]}```")
        return None
    if state["state"] == "cancelled":
        await message.edit(content=f"⏹ المهمة #{job_id} اتوقفت.")
        return None
    await message.edit(content=f"✅ المهمة #{job_id} · {KIND_LABELS[kind]} · {info['pages']} صفحة · بجهّز الناتج...")
    return info


def describe(state: dict, pages: int, target: str) -> str:
    kind = state["state"]
    if kind == "queued":
        return f"{pages} صفحة → `{target}` · في الانتظار"
    if kind == "running":
        steps, finished = state.get("steps") or 0, state.get("finished") or 0
        progress = f"{finished}/{steps} خطوة" if steps else "بيحمّل النماذج (أول مرة بتاخد دقايق)"
        return f"{progress} · {state.get('current') or ''}".strip(" ·")
    if kind == "rendering":
        return "بيرسم الصفحات النهائية..."
    if kind == "done":
        return f"منتهية · {pages} صفحة → `{target}`"
    return kind


async def deliver(channel, job_id: int, info: dict, kind: str, output: str, overlay: bool = False):
    """Sends a done job's output to `channel` in the shape asked for."""
    pages = info["status"]["pages"]
    count = info["pages"]
    if kind == "detect" or output == "overlay":
        await send_overlays(channel, job_id, pages)
        counts = sum(len(p["blocks"]) for p in pages), sum(len(p["balloons"]) for p in pages), sum(len(p["panels"]) for p in pages)
        await channel.send(f"🔎 المهمة #{job_id}: {counts[0]} نص · {counts[1]} فقاعة · {counts[2]} لوحة في {count} صفحة.")
        return
    if kind == "ocr":
        await send_text(channel, job_id, pages, translated=False)
        if overlay:
            await send_overlays(channel, job_id, pages)
        return
    if output == "text":
        await send_text(channel, job_id, pages, translated=True)
        return
    if output == "cleaned":
        await send_pages(channel, job_id, count, suffix="/cleaned")
        return
    if output == "cbz":
        data = await fetch(f"/jobs/{job_id}/cbz")
        if data is not None and len(data) <= FILE_BYTES:
            await channel.send(file=discord.File(io.BytesIO(data), filename=f"koharu-{job_id}.cbz"))
        else:
            size = f"{len(data) / 1e6:.0f} MB" if data else "غير متاح"
            await channel.send(f"الـ CBZ أكبر من حد Discord ({size})؛ ببعت الصفحات فرادى.")
            await send_pages(channel, job_id, count)
    elif output == "psd":
        data = await fetch(f"/jobs/{job_id}/psd.zip")
        if data is not None and len(data) <= FILE_BYTES:
            await channel.send(file=discord.File(io.BytesIO(data), filename=f"koharu-{job_id}-psd.zip"))
        else:
            await channel.send("ملف الـ PSD المجمّع أكبر من حد Discord؛ بحاول صفحة صفحة.")
            for number in range(1, count + 1):
                psd = await fetch(f"/jobs/{job_id}/pages/{number}/psd")
                if psd is not None and len(psd) <= FILE_BYTES:
                    await channel.send(file=discord.File(io.BytesIO(psd), filename=f"{number:03}.psd"))
                else:
                    await channel.send(f"صفحة {number}: الـ PSD أكبر من حد Discord ({(len(psd) if psd else 0) / 1e6:.0f} MB).")
    else:
        await send_pages(channel, job_id, count)
    if kind == "translate":
        await send_text(channel, job_id, pages, translated=True)
    await channel.send(f"✅ المهمة #{job_id} اتبعتت. `/export job:{job_id}` لشكل آخر، `/redo job:{job_id}` لإعادة الترجمة، `/ask job:{job_id}` للوكيل.")


async def send_pages(channel, job_id: int, count: int, suffix: str = ""):
    files = []
    for number in range(1, count + 1):
        data = await fetch(f"/jobs/{job_id}/pages/{number}{suffix}")
        if data is None:
            await channel.send(f"صفحة {number}: مش متاحة.")
            continue
        files.append(as_discord_file(data, f"{number:03}.png"))
        if len(files) == FILES_PER_MESSAGE:
            await channel.send(files=files)
            files = []
    if files:
        await channel.send(files=files)


async def send_overlays(channel, job_id: int, pages: list):
    files = []
    for page in pages:
        data = await fetch(f"/jobs/{job_id}/pages/{page['page']}")
        if data is None:
            continue
        files.append(as_discord_file(draw_overlay(data, page), f"{page['page']:03}-boxes.png"))
        if len(files) == FILES_PER_MESSAGE:
            await channel.send(files=files)
            files = []
    if files:
        await channel.send(files=files)


async def send_text(channel, job_id: int, pages: list, translated: bool):
    summary = texts_summary(pages, translated)
    if not summary:
        await channel.send("مفيش نصوص مكتشفة.")
        return
    name = f"job-{job_id}-{'translation' if translated else 'ocr'}.txt"
    if len(summary) < 1800:
        await channel.send(f"📝 نصوص المهمة #{job_id}:\n```\n{summary}\n```")
    else:
        await channel.send(content=f"📝 نصوص المهمة #{job_id}:",
                           file=discord.File(io.BytesIO(summary.encode("utf-8")), filename=name))


def texts_summary(pages: list, translated: bool) -> str:
    lines = []
    for page in pages:
        blocks = [b for b in page["blocks"] if b.get("source", "").strip() or b.get("translation")]
        if not blocks:
            continue
        lines.append(f"== page {page['page']} ==")
        for number, block in enumerate(page["blocks"], 1):
            source = block.get("source", "").replace("\n", " ")
            tag = " [SFX]" if block.get("role") == "sfx" else ""
            if translated:
                target = (block.get("translation") or "—").replace("\n", " ")
                lines.append(f"{number:02}.{tag} {source}  →  {target}")
            else:
                lines.append(f"{number:02}.{tag} {source}")
        lines.append("")
    return "\n".join(lines).strip()


def draw_overlay(png: bytes, page: dict) -> bytes:
    """The page with its panels (grey), balloons (green), and texts (blue, sound effects orange) boxed and numbered."""
    image = Image.open(io.BytesIO(png)).convert("RGB")
    draw = ImageDraw.Draw(image)
    width = image.size[0]
    stroke = max(2, width // 500)
    try:
        font = ImageFont.load_default(size=max(16, width // 45))
    except TypeError:  # Pillow before 10.1 has no sized default font
        font = ImageFont.load_default()

    def box(bbox):
        x0, y0, x1, y1 = (int(round(v)) for v in bbox)
        return [min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)]

    for bbox in page.get("panels", []):
        draw.rectangle(box(bbox), outline=(128, 128, 128), width=stroke)
    for bbox in page.get("balloons", []):
        draw.rectangle(box(bbox), outline=(0, 170, 90), width=stroke)
    for number, block in enumerate(page.get("blocks", []), 1):
        if not block.get("bbox"):
            continue
        color = (255, 140, 0) if block.get("role") == "sfx" else (30, 110, 255)
        x0, y0, x1, y1 = box(block["bbox"])
        draw.rectangle([x0, y0, x1, y1], outline=color, width=stroke)
        label = str(number)
        left, top, right, bottom = draw.textbbox((0, 0), label, font=font)
        tw, th = right - left, bottom - top
        ly = y0 - th - 6 if y0 - th - 6 >= 0 else y0
        draw.rectangle([x0, ly, x0 + tw + 8, ly + th + 6], fill=color)
        draw.text((x0 + 4 - left, ly + 3 - top), label, fill="white", font=font)
    out = io.BytesIO()
    image.save(out, "PNG", optimize=True)
    return out.getvalue()


def as_discord_file(data: bytes, name: str) -> discord.File:
    """The PNG as is, or re-encoded as JPEG when Discord wouldn't take it."""
    if len(data) <= FILE_BYTES:
        return discord.File(io.BytesIO(data), filename=name)
    image = Image.open(io.BytesIO(data)).convert("RGB")
    out = io.BytesIO()
    for quality in (92, 85, 75, 60):
        out = io.BytesIO()
        image.save(out, "JPEG", quality=quality, optimize=True)
        if out.tell() <= FILE_BYTES:
            break
    out.seek(0)
    return discord.File(out, filename=name.rsplit(".", 1)[0] + ".jpg")


if __name__ == "__main__":
    bot.run(os.environ["DISCORD_TOKEN"])
