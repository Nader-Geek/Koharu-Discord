# Koharu-Discord

بوت Discord لترجمة المانجا وتنظيفها بـ [Koharu](https://github.com/koharu-rs/koharu)، مع worker بدون نافذة يشتغل على Google Colab أو أي جهاز Linux بـ GPU.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Nader-Geek/Koharu-Discord/blob/main/koharu_discord_colab.ipynb)

## الخلاصة أولًا

- **ده مش تشغيل لتطبيق Koharu نفسه.** Koharu تطبيق سطح مكتب (Tauri + CEF + نافذة GPU) وخادم الـ MCP بتاعه بيشتغل بس جوه التطبيق. اللي هنا هو **worker بدون نافذة** (`koharu-headless`) مبني من مكتبات Koharu نفسها (المشروع، الـ pipeline، الراسم، تصدير PSD، والوكيل)، وبوت Discord بـ Python بيكلمه عبر HTTP محلي. الريبو الأصلي فيه أمثلة headless بنفس الفكرة (`koharu-pipeline/examples/run.rs` و`koharu-agent/examples/serve.rs`)، والـ worker مبني على نفس الـ APIs، فالناتج مطابق لتصدير التطبيق.
- **Colab فيه عقبة واحدة حقيقية:** تعريف NVIDIA الحالي على Colab هو 580 (CUDA 13.0)، وKoharu بيرفض أي تعريف أقدم من 13.3. رقعة صغيرة هنا بتضيف متغير بيئة `KOHARU_MIN_CUDA_DRIVER` لتخطي الفحص اعتمادًا على توافق الإصدارات الفرعية في CUDA. ده **تجريبي وغير مُختبر**: لو نواة ما في libtorch أو llama.cpp مترجمة PTX فقط، التعريف الأقدم هيفشل في تشغيلها.
- **حالة الكود:** الـ worker مكتوب على التوقيعات الفعلية في مستودع Koharu عند الـ commit المثبّت في `worker/UPSTREAM_COMMIT`، لكنه **لم يُجمَّع ولم يُشغَّل بعد**؛ أول `cargo build` (من الـ workflow هنا) هو الاختبار الحقيقي، وتوقّع تصحيحات صغيرة. البوت اتفحص بمفسّر Python فقط.

## محتويات الريبو

| المسار | الوظيفة |
|---|---|
| `bot.py` | بوت Discord بكل الأوامر تحت |
| `koharu_discord_colab.ipynb` | النوتبوك الكامل: فحص البيئة، حزم النظام، تشغيل الـ worker، تسخين، اختبار، ثم البوت |
| `make_notebook.py` | يولّد النوتبوك من `bot.py`؛ شغّله بعد أي تعديل على البوت |
| `requirements.txt` | discord.py وaiohttp وPillow |
| `worker/koharu-headless/` | الـ crate: HTTP API على `127.0.0.1:8787` فوق الـ pipeline والراسم وتصدير PSD والوكيل |
| `worker/patches/` | رقعة `KOHARU_MIN_CUDA_DRIVER` على `koharu-runtime` |
| `worker/UPSTREAM_COMMIT` | commit الـ Koharu المبني عليه |
| `worker/build.sh` | يستنسخ Koharu، يضيف الـ crate، يطبّق الرقع، ويبني |
| `.github/workflows/build-headless.yml` | يبني الـ binary على Ubuntu 24.04 ويرفعه كـ artifact، وكـ Release عند دفع tag |

## أوامر البوت

| الأمر | المراحل | الناتج |
|---|---|---|
| `/translate` | كشف → OCR → تنظيف → ترجمة (`cleanup:false` يكتب الترجمة فوق الأصل بدون تنظيف) | صور الصفحات / CBZ / PSD بطبقات / نص، + ملف المصدر→الترجمة |
| `/clean` | كشف → تنظيف | الصفحات بدون نص (LaMa افتراضي، أو FLUX.3 Klein / Qwen-Image) |
| `/ocr` | كشف → OCR | ملف نصي بترتيب القراءة، وصورة مرقّمة اختياريًا |
| `/detect` | كشف | صورة بالصناديق: نصوص أزرق، SFX برتقالي، فقاعات أخضر، لوحات رمادي، مع العدد |
| كليك يمين → Apps | **Translate with Koharu** / **Clean with Koharu** | على أي رسالة فيها صور أو أرشيف |
| `/redo job action` | ترجمة فقط / OCR+ترجمة / تنظيف فقط / الكل، بلغة أو نموذج أو تعليمات جديدة | نفس المشروع بدون رفع الصفحات تاني |
| `/export job format` | بدون معالجة | الصفحات النهائية، **الصفحات المنظّفة فقط** (لمن يريد التنضيد بنفسه)، CBZ، PSD، النصوص، صورة الكشف |
| `/ask job prompt` | وكيل Koharu بأدواته الثمانية (قراءة، تعديل، تشغيل، عرض) | رده + الأدوات التي استخدمها؛ الصفحات تُعاد رسمها بعد أي تعديل. يتطلب `--agent-model` |
| `/status [job]` `/cancel job` `/models` `/help` | | |
| `/settings target\|model\|instructions\|show\|reset` | | افتراضيات السيرفر (Manage Server) |
| `/glossary add\|remove\|list\|clear` | | قاموس أسماء ومصطلحات يُلحق بتعليمات كل ترجمة |
| `/worker status\|warmup\|unload\|translator\|recognizer\|inpainter` | | إدارة الـ worker (Manage Server) |

كل مهمة بتظهر برسالة تقدم في القناة (الخطوات المنتهية، المرحلة والصفحة الحالية)، لأن token التفاعل في Discord بينتهي بعد 15 دقيقة. الملفات أكبر من حد Discord (10 MB) بتتحول JPEG تلقائيًا للصور، وبتتقسم أو تتبعت صفحة صفحة للـ CBZ/PSD.

## المعمارية

```
Discord ──(Gateway)──▶ bot.py (discord.py) ──HTTP 127.0.0.1:8787──▶ koharu-headless
                                                                          │
                                                    koharu-scene     مشروع .khr لكل مهمة (يبقى مفتوحًا)
                                                    koharu-pipeline  كشف → OCR → ترميم → ترجمة
                                                    koharu-renderer  PNG لكل صفحة على vello/wgpu
                                                    koharu-psd       PSD بطبقات
                                                    koharu-agent     أدوات الوكيل الثمانية على المشروع
                                                                          │
                                                    koharu-runtime ──▶ libtorch / llama.cpp / sd.cpp (CUDA)
```

- المهام بتتنفذ واحدة واحدة، والنماذج فاضلة محمّلة بينها. اختيار نموذج في مهمة بيصبح الافتراضي للمهام اللي بعدها (نموذج واحد لكل مرحلة في الذاكرة).
- "الصفحات المنظّفة" بتتحسب زي مثال الريبو الأصلي: المسح + طبقات التنظيف عبر أقنعتها، بدون أي نص.
- مفيش Sentry في الـ worker.

### الـ API

| الطلب | الرد |
|---|---|
| `GET /health` | `ok` |
| `POST /warmup` | ينزّل ويحمّل الـ runtime للـ GPU ويرجّع اسمه وذاكرته |
| `GET /models` | نماذج الترجمة (محلية + أي مزود له مفتاح)، نماذج OCR والترميم، والمستخدم حاليًا |
| `POST /config` `{translator?, recognizer?, inpainter?}` | يغيّر الافتراضيات |
| `POST /unload` | يفرّغ كل النماذج |
| `GET /jobs` | كل المهام، الأحدث أولًا |
| `POST /jobs` (multipart: `pages`×N، `target`، `stages`، `instructions`، `translator`/`recognizer`/`inpainter`) | `{"id", "pages"}` |
| `GET /jobs/{id}` | `status.state` = queued / running (`finished`/`steps`/`current`) / rendering / done (مع `pages`: اللوحات والفقاعات والكتل بصناديقها ونصوصها) / failed / cancelled |
| `POST /jobs/{id}/rerun` `{stages?, target?, instructions?, translator?, ...}` | يعيد المراحل على نفس المشروع |
| `POST /jobs/{id}/cancel` | يوقف مهمة في الانتظار أو التشغيل |
| `POST /jobs/{id}/ask` `{prompt, reasoning?}` | `{answer, calls}` |
| `GET /jobs/{id}/pages/{n}` · `/cleaned` · `/psd` | PNG نهائي، PNG منظّف، PSD |
| `GET /jobs/{id}/cbz` · `/psd.zip` | أرشيفات كل الصفحات |
| `DELETE /jobs/{id}` | يغلق مشروع مهمة منتهية ويحذف ملفاتها |

## خطوات التشغيل

### 1) ابنِ الـ binary مرة واحدة

ادفع tag في الريبو ده:

```bash
git tag v1 && git push origin v1
```

الـ workflow **Build headless** بيبني `koharu-headless` لـ Linux x86-64 على Ubuntu 24.04 (نحو 30 إلى 60 دقيقة أول مرة) وبيعمل Release فيه الملف. النوتبوك بينزّله من `releases/latest/download/koharu-headless`. أي push يلمس `worker/` بيبني كمان ويحفظ الناتج كـ artifact في صفحة الـ run. للبناء محليًا أو على fork من Koharu، شوف `worker/README.md`.

**لو الريبو خاص:** رابط التنزيل العام ما بيشتغلش. أضف في Colab Secret باسم `GITHUB_TOKEN` (Fine-grained token على الريبو ده بصلاحية Contents: Read) والنوتبوك بينزّل الملف عبر الـ API تلقائيًا. زر Open in Colab كمان بيحتاج تربط Colab بحساب GitHub بتاعك. البديل الأبسط: اجعل الريبو عامًا.

### 2) Discord Developer Portal

New Application → Bot → Reset Token → انسخه. من OAuth2 → URL Generator: scopes `bot` + `applications.commands`، وصلاحيات Send Messages وAttach Files وRead Message History. مش محتاج Message Content Intent.

### 3) Colab

افتح النوتبوك بزر Open in Colab فوق، اختار T4، حط `DISCORD_TOKEN` في Secrets، ونفّذ الخلايا بالترتيب. الخلية 6 بتختبر صفحة واحدة (نهائية ومنظّفة) قبل ما تشغّل البوت، عشان تفصل مشاكل الـ GPU عن مشاكل Discord. لتفعيل `/ask` حط `AGENT_MODEL` في الخلية 4.

### على جهازك بدل Colab

```bash
./out/koharu-headless --listen 127.0.0.1:8787 --translator koharu/index-translate-2b &
pip install -r requirements.txt
DISCORD_TOKEN=... KOHARU_TARGET=ar python bot.py
```

## قيود لازم تعرفها

1. **التعريف:** لو ظهر في الـ log `CUDA driver is too old` من غير المتغير، أو أخطاء `no kernel image is available` / `unsupported PTX version` مع المتغير، يبقى تعريف Colab مش كفاية لحزم CUDA 13.3. الحل وقتها جهاز بتعريف R610 أو أحدث. Vulkan مش بديل عملي على Colab: مفيش ICD لـ NVIDIA في الحاوية، وlibtorch على Vulkan بيشتغل على الـ CPU أصلًا.
2. **الراسم محتاج GPU عبر wgpu:** الحاوية من غير Vulkan ICD، فالنوتبوك بيثبّت `mesa-vulkan-drivers` (lavapipe) عشان الرسم يشتغل على الـ CPU.
3. **ذاكرة T4 (15 GB):** Koharu ما بيفرّغش نموذج عشان يفسح مكان؛ النموذج اللي ما يسعش بيفشل برسالة فيها الذاكرة الحرة. ابدأ بـ Index-Translate 2B + Hayai + LaMa. الوكيل بيحمّل نموذجًا إضافيًا فوقهم، فابدأ بـ 2B-4B أو سيبه معطلًا. ماتختارش FLUX.3/Qwen-Image على T4.
4. **الوكيل والـ `/redo` على نفس المشروع:** مشروع المهمة بيفضل مفتوحًا على الـ worker لحد ما يتحذف أو يتجاوز عدد المهام `--max-jobs` (40 افتراضيًا)؛ بعدها الأقدم بيتقفل وملفاته بتتحذف. إعادة تشغيل الـ worker بتمسح كل المهام.
5. **جلسة Colab:** الحد الأقصى 12 ساعة، وبتتقفل مع الخمول في الطبقة المجانية، والتبويب لازم يفضل مفتوح. مناسب للتجارب والاستخدام الشخصي، مش لبوت 24/7.
6. **مزودون مستضافون (OpenAI وغيره):** مفاتيحهم بتتخزن في `secrets.age` بمفتاح في مخزن اعتماد النظام، وColab مفيهوش Secret Service. الأسهل النماذج المحلية.
7. **حدود Discord:** 10 ملفات لكل رسالة و10 MB للملف؛ البوت بيتعامل معها زي ما هو موضح فوق. الـ PSD بطبقات غالبًا بيتعدى 10 MB للصفحة الواحدة؛ وقتها حمّله من Colab مباشرة (`/content/koharu-jobs/<id>/`) أو ارفع حد السيرفر بالـ boost.
8. **مدة المعالجة:** على T4 الصفحة من نصف دقيقة لدقائق حسب عدد الفقاعات والنموذج. أول مهمة بتتأخر زيادة لتنزيل النماذج. `/ask` بيبدأ بتحميل نموذج الوكيل أول مرة.
9. **إعادة الترجمة (`/redo action:translate`)** بتحتاج مهمة فيها OCR أصلًا (ترجمة أو OCR)؛ على مهمة `/clean` هترجع بلا نصوص لأن المرحلة "بلا عمل ليست فشلًا" في Koharu. استخدم `OCR + ترجمة` وقتها.

## استكشاف الأخطاء

| الرسالة | المعنى | الحل |
|---|---|---|
| `GPU is not available` | كل الـ probes رفضت الجهاز | شوف الـ log؛ غالبًا `CUDA driver is too old` → `KOHARU_MIN_CUDA_DRIVER=13000` |
| `couldn't load .../libtorch_cuda.so` | مكتبة ما اتحملتش أو glibc أقدم | تأكد من Ubuntu 24.04 (glibc 2.39)؛ أعد `POST /warmup` |
| `no kernel image is available` / `PTX ... unsupported toolchain` | التعريف أقدم مما تحتاجه النوى | جهاز بتعريف R610+ |
| `no GPU adapter is available` | wgpu ما لقاش Vulkan/GL | `apt install mesa-vulkan-drivers libvulkan1 libegl1 libgl1-mesa-dri` |
| `couldn't load <model> with X free on ...` | الذاكرة لا تكفي | نموذج أصغر، أو `/worker unload` قبل المهمة |
| `llama.cpp doesn't list CUDA0 at 0` | llama.cpp ما شافش الـ GPU | تأكد من `nvidia-smi`؛ أعد تشغيل الـ worker |
| `CLDR has no English name for ...` | رمز لغة غير معروف | استخدم رموز BCP 47 قياسية: `ar`، `en`، `zh-Hans`... |
| `this worker has no agent model` | `/ask` بدون `--agent-model` | حط `AGENT_MODEL` في خلية تشغيل الـ worker |
| `the model says nothing of its limits` / النموذج لا يدعم الأدوات | نموذج الوكيل لا يستدعي أدوات (مثل Index-Translate) | اختر نموذجًا عامًا: Qwen 3.5/3.6، Gemma 4، LFM3 |

## البدائل لو Colab ما نفعش

- أي VM بـ GPU وUbuntu 24.04 وتعريف ≥ R610 (RunPod / Vast / Lightning وغيرها): نفس الـ binary ونفس `bot.py` بدون أي تعديل، والبوت يقدر يفضل شغال 24/7.
- على جهاز فيه Koharu نفسه شغال، تقدر تستخدم `cargo run -p koharu-agent --example serve` (خادم MCP بدون نافذة) لو عايز البوت يستخدم أدوات الوكيل بدل الـ pipeline المباشر، لكن الاستيراد والتصدير هيفضلوا على عاتقك.

## الرخص

كود الـ worker وبوت Discord في هذا الريبو ملك صاحبه. الـ crate يُبنى داخل مستودع Koharu ويعتمد على مكتباته المرخّصة MIT OR Apache-2.0، وبعض نماذج Koharu لها رخص خاصة (manga-ocr-ar غير تجاري، Qwen-Image للبحث فقط)؛ راجع README الأصلي قبل الاستخدام التجاري.
