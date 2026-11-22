# koharu-headless

نسخة بدون نافذة من Koharu: الـ pipeline (كشف → OCR → ترميم → ترجمة)، الراسم، تصدير PSD، والوكيل، وراء HTTP API محلي على `127.0.0.1:8787`. البوت في جذر الريبو بيكلمها.

## المحتويات

| المسار | الوظيفة |
|---|---|
| `koharu-headless/` | الـ crate كما يُضاف إلى `crates/` في مستودع Koharu |
| `patches/0001-cuda-driver-override.patch` | يضيف متغير البيئة `KOHARU_MIN_CUDA_DRIVER` لتخطي الحد الأدنى لتعريف CUDA (Colab على 13.0 وKoharu يطلب 13.3) |
| `UPSTREAM_COMMIT` | commit الـ Koharu الذي كُتب عليه الكود ويُبنى منه |
| `build.sh` | يستنسخ Koharu عند ذلك الـ commit، يضيف الـ crate، يطبّق الرقع، ويبني بـ `cargo build --release -p koharu-headless` |

## البناء

على GitHub: Actions → **Build headless** → Run workflow (أو ادفع tag مثل `v1` فيُنشأ Release بالملف).

محليًا على Linux x86-64 (Ubuntu 24.04 أو أحدث، ومعه `clang libclang-dev llvm pkg-config libssl-dev` وRust 1.97.1+):

```bash
bash worker/build.sh
./out/koharu-headless --help
```

على fork خاص بك من Koharu بدل السكربت:

```bash
cp -R worker/koharu-headless /path/to/koharu/crates/
git -C /path/to/koharu apply "$PWD"/worker/patches/*.patch
cargo build --release -p koharu-headless
```

## تحديث Koharu

غيّر الـ commit في `UPSTREAM_COMMIT`، وأعد البناء. لو تغيّرت واجهات `koharu-scene` أو `koharu-pipeline` أو `koharu-renderer` أو `koharu-agent`، عدّل `koharu-headless/src/main.rs` تبعًا لها.

## التشغيل

```bash
./out/koharu-headless --listen 127.0.0.1:8787 --jobs koharu-jobs \
  --translator koharu/index-translate-2b --recognizer hayai-ocr --inpainter lama \
  --agent-model koharu/qwen3.5-4b \
  --fonts "Noto Naskh Arabic,Noto Sans Arabic,Noto Sans CJK JP"
```

الـ API كاملة في README الجذر. لا توجد مصادقة، فاتركه على `127.0.0.1`.
