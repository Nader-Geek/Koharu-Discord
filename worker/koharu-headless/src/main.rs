//! Koharu without a window: the pipeline, the renderer, the PSD export, and
//! the agent behind a small HTTP API on this machine, for a bot or a server
//! to drive. One job is one project: its uploaded images become pages, the
//! stages run over them, and each page renders to a PNG as the app would
//! export it. A job's project stays open afterwards, so its pages can be run
//! again, exported as PSD, or worked on by the agent. Models stay loaded
//! between jobs, as they do in the app.
//!
//! `koharu-headless --listen 127.0.0.1:8787 --translator koharu/index-translate-2b`
//!
//! | Route | Does |
//! |---|---|
//! | `GET /health` | answers `ok` once the server listens |
//! | `POST /warmup` | downloads and loads the native runtime for this GPU |
//! | `GET /models` | the models each stage can use, and the ones in use |
//! | `POST /config` | JSON `translator`, `recognizer`, `inpainter`: the models every later job uses |
//! | `POST /unload` | unloads every model |
//! | `GET /jobs` | every job, newest first |
//! | `POST /jobs` | multipart: `pages` files (images, ZIP, or CBZ), `target`, `stages`, `instructions`, and a model per stage; answers the job's id |
//! | `GET /jobs/{id}` | the job's state, and its pages' layout and texts once done |
//! | `POST /jobs/{id}/rerun` | JSON `stages`, `target`, `instructions`, models: runs stages again on the same project |
//! | `POST /jobs/{id}/cancel` | stops a queued or running job |
//! | `POST /jobs/{id}/ask` | JSON `prompt`, `reasoning`: a turn of the agent on the project, with `--agent-model` |
//! | `GET /jobs/{id}/pages/{n}` | page `n` (from 1) as the app exports it, PNG |
//! | `GET /jobs/{id}/pages/{n}/cleaned` | page `n` with its clean-ups and no lettering, PNG |
//! | `GET /jobs/{id}/pages/{n}/psd` | page `n` as a layered PSD |
//! | `GET /jobs/{id}/cbz` | every page as the app exports it, in a CBZ |
//! | `GET /jobs/{id}/psd.zip` | every page as a PSD, zipped |
//! | `DELETE /jobs/{id}` | closes a finished job and removes its files |

use std::{
    collections::{BTreeMap, BTreeSet},
    io::{Cursor, Read as _, Write as _},
    net::SocketAddr,
    path::{Path, PathBuf},
    sync::{
        Arc,
        atomic::{AtomicBool, AtomicU64, Ordering},
    },
    time::Duration,
};

use anyhow::{Context as _, Result};
use axum::{
    Json, Router,
    extract::{DefaultBodyLimit, Multipart, Path as UrlPath, State, multipart::MultipartError},
    http::{StatusCode, header},
    response::{IntoResponse, Response},
    routing::{get, post},
};
use clap::Parser;
use image::{Rgba, RgbaImage};
use koharu_agent::{Agent, AgentConfig, Event as AgentEvent, Tools};
use koharu_llm::{
    Context as LlmContext, Size, SummaryPrompts,
    rig::{driver::DynModel, operation::Completion},
};
use koharu_pipeline::{
    Event, Inpainter, Outcome, Pipeline, PipelineConfig, Recognizer, Request, Stage, Target,
};
use koharu_psd::Labels;
use koharu_renderer::{Renderer, RendererConfig};
use koharu_scene::{
    Content, Edit, Hash, LanguageIdentifier, Layer, Op, Page, PageId, Project, Role, Session,
    Tile,
};
use kurbo::Shape as _;
use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use strum::VariantArray;
use tokio::sync::Semaphore;
use tracing_subscriber::EnvFilter;

#[derive(Debug, Parser)]
#[command(
    version,
    about = "Koharu's pipeline, renderer, PSD export, and agent over HTTP, with no window"
)]
struct Cli {
    /// The address to listen on. Keep it on this machine: there is no
    /// authentication.
    #[arg(long, default_value = "127.0.0.1:8787")]
    listen: SocketAddr,

    /// Where each job's project file and PNGs go, in a folder named by the
    /// job's id; a folder a previous run left under an id goes when the id
    /// is used again.
    #[arg(long, default_value = "koharu-jobs")]
    jobs: PathBuf,

    /// The translation model, a `vendor/model` reference, with an optional
    /// quantization after a colon.
    #[arg(long, default_value = "koharu/index-translate-2b")]
    translator: String,

    /// The OCR model: hayai-ocr, manga-ocr-ar, or baberu-ocr.
    #[arg(long, default_value = "hayai-ocr")]
    recognizer: String,

    /// The inpainting model: lama, flux3-klein, or qwen-image.
    #[arg(long, default_value = "lama")]
    inpainter: String,

    /// The agent's model, a `vendor/model` reference to one that calls
    /// tools, such as koharu/qwen3.5-4b; without one, `ask` is off.
    #[arg(long)]
    agent_model: Option<String>,

    /// The preferred font families, in order, for texts with no font of their
    /// own and for characters a font lacks.
    #[arg(
        long,
        value_delimiter = ',',
        default_value = "Noto Naskh Arabic,Noto Sans Arabic,Noto Sans CJK JP"
    )]
    fonts: Vec<String>,

    /// The most mebibytes one upload may hold.
    #[arg(long, default_value_t = 256)]
    max_upload_mib: usize,

    /// How many jobs to keep open; past it, the oldest finished ones close
    /// and their files go.
    #[arg(long, default_value_t = 40)]
    max_jobs: usize,
}

/// A value by its id, as the `[pipeline]` section and requests write it.
fn parse<T: serde::de::DeserializeOwned>(id: &str) -> Result<T> {
    serde_json::from_value(Value::String(id.to_owned()))
        .map_err(|error| anyhow::anyhow!("`{id}`: {error}"))
}

struct App {
    pipeline: Pipeline,
    renderer: Renderer,
    /// The models every job uses, which a job's own choice replaces for the
    /// jobs after it too, so one model of each stage is loaded at a time.
    config: Mutex<PipelineConfig>,
    agent_model: Option<String>,
    /// The agent's model, connected on the first ask and kept loaded.
    agent: tokio::sync::Mutex<Option<DynModel<Completion>>>,
    jobs_dir: PathBuf,
    jobs: Mutex<BTreeMap<u64, Job>>,
    next: AtomicU64,
    /// One run at a time, so a chapter comes back in order and the GPU's
    /// memory stays predictable.
    turn: Semaphore,
    max_jobs: usize,
}

struct Job {
    dir: PathBuf,
    pages: usize,
    stages: BTreeSet<Stage>,
    target: LanguageIdentifier,
    instructions: String,
    /// The job's project, open from its import until the job is deleted.
    session: Option<Arc<Mutex<Session>>>,
    /// The agent's conversation about the project, from its first ask.
    agent: Option<Arc<tokio::sync::Mutex<Agent>>>,
    /// One thing at a time on this job: its run, a rerun, an ask, or an
    /// export.
    busy: Arc<tokio::sync::Mutex<()>>,
    /// Set by a cancel while the job waits its turn.
    cancelled: Arc<AtomicBool>,
    state: JobState,
}

#[derive(Clone, Serialize)]
#[serde(tag = "state", rename_all = "snake_case")]
enum JobState {
    Queued,
    Running {
        /// Page stages finished, of `steps`; 0 of 0 while the run hasn't
        /// been queued yet, as when its models are still loading.
        finished: usize,
        steps: usize,
        /// The page stage the GPU took last.
        current: Option<String>,
    },
    Rendering,
    Done {
        pages: Vec<PageInfo>,
    },
    Failed {
        error: String,
    },
    Cancelled,
}

impl JobState {
    fn kind(&self) -> &'static str {
        match self {
            Self::Queued => "queued",
            Self::Running { .. } => "running",
            Self::Rendering => "rendering",
            Self::Done { .. } => "done",
            Self::Failed { .. } => "failed",
            Self::Cancelled => "cancelled",
        }
    }

    /// Whether the job is over, one way or another.
    fn settled(&self) -> bool {
        matches!(
            self,
            Self::Done { .. } | Self::Failed { .. } | Self::Cancelled
        )
    }
}

fn running(finished: usize, steps: usize, current: Option<String>) -> JobState {
    JobState::Running {
        finished,
        steps,
        current,
    }
}

/// A page's layout and texts, in reading order, with every box in page
/// pixels as `[x0, y0, x1, y1]`.
#[derive(Clone, Serialize)]
struct PageInfo {
    page: usize,
    width: u32,
    height: u32,
    panels: Vec<[f64; 4]>,
    balloons: Vec<[f64; 4]>,
    blocks: Vec<BlockInfo>,
}

#[derive(Clone, Serialize)]
struct BlockInfo {
    /// `text`, or `sfx` for a sound effect.
    role: &'static str,
    bbox: Option<[f64; 4]>,
    in_balloon: bool,
    visible: bool,
    source: String,
    translation: Option<String>,
}

/// The models a job or a config change names, each by its id.
#[derive(Clone, Debug, Default, Deserialize)]
#[serde(default, deny_unknown_fields)]
struct Models {
    translator: Option<String>,
    recognizer: Option<String>,
    inpainter: Option<String>,
}

/// What a run asks for.
struct Options {
    stages: BTreeSet<Stage>,
    target: LanguageIdentifier,
    instructions: String,
    models: Models,
}

#[derive(Default, Deserialize)]
#[serde(default, deny_unknown_fields)]
struct RerunRequest {
    /// Comma-separated; the job's own stages when unset.
    stages: Option<String>,
    target: Option<String>,
    instructions: Option<String>,
    translator: Option<String>,
    recognizer: Option<String>,
    inpainter: Option<String>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct AskRequest {
    prompt: String,
    /// `off`, or an effort level such as `low` or `high`; the model's own
    /// when unset.
    #[serde(default)]
    reasoning: Option<String>,
}

/// A failed request, as its status and message.
struct ApiError(StatusCode, String);

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        (self.0, self.1).into_response()
    }
}

impl From<anyhow::Error> for ApiError {
    fn from(error: anyhow::Error) -> Self {
        Self(StatusCode::INTERNAL_SERVER_ERROR, format!("{error:#}"))
    }
}

impl From<MultipartError> for ApiError {
    fn from(error: MultipartError) -> Self {
        Self(StatusCode::BAD_REQUEST, error.to_string())
    }
}

fn bad(message: impl Into<String>) -> ApiError {
    ApiError(StatusCode::BAD_REQUEST, message.into())
}

fn conflict(message: impl Into<String>) -> ApiError {
    ApiError(StatusCode::CONFLICT, message.into())
}

fn not_found(id: u64) -> ApiError {
    ApiError(StatusCode::NOT_FOUND, format!("there is no job {id}"))
}

#[tokio::main]
async fn main() -> Result<()> {
    let filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info"));
    tracing_subscriber::fmt().with_env_filter(filter).init();
    let cli = Cli::parse();
    fs_err::create_dir_all(&cli.jobs)?;

    let mut config = PipelineConfig::default();
    config.translator.model = cli.translator;
    config.recognizer.model = parse(&cli.recognizer)?;
    config.inpainter.model = parse(&cli.inpainter)?;
    // Needs the Tokio runtime: the scheduler and the GPU thread start here,
    // and models load the first time a stage needs them.
    let pipeline = Pipeline::new(config.clone());
    let renderer = Renderer::new(RendererConfig {
        font_families: cli.fonts,
    });
    let app = Arc::new(App {
        pipeline,
        renderer,
        config: Mutex::new(config),
        agent_model: cli.agent_model,
        agent: tokio::sync::Mutex::new(None),
        jobs_dir: cli.jobs,
        jobs: Mutex::new(BTreeMap::new()),
        next: AtomicU64::new(0),
        turn: Semaphore::new(1),
        max_jobs: cli.max_jobs.max(1),
    });

    let router = Router::new()
        .route("/health", get(|| async { "ok" }))
        .route("/warmup", post(warmup))
        .route("/unload", post(unload))
        .route("/models", get(models))
        .route("/config", post(set_config))
        .route("/jobs", get(list_jobs).post(create_job))
        .route("/jobs/{id}", get(get_job).delete(delete_job))
        .route("/jobs/{id}/rerun", post(rerun_job))
        .route("/jobs/{id}/cancel", post(cancel_job))
        .route("/jobs/{id}/ask", post(ask))
        .route("/jobs/{id}/cbz", get(get_cbz))
        .route("/jobs/{id}/psd.zip", get(get_psd_zip))
        .route("/jobs/{id}/pages/{page}", get(get_page))
        .route("/jobs/{id}/pages/{page}/cleaned", get(get_cleaned))
        .route("/jobs/{id}/pages/{page}/psd", get(get_psd))
        .layer(DefaultBodyLimit::max(cli.max_upload_mib << 20))
        .with_state(app);
    let listener = tokio::net::TcpListener::bind(cli.listen).await?;
    tracing::info!(
        "koharu-headless listens on http://{}",
        listener.local_addr()?
    );
    axum::serve(listener, router).await?;
    Ok(())
}

// --- The worker ---

/// Downloads and loads the native runtime for this machine's GPU, which the
/// first job would otherwise wait for, and names the device.
async fn warmup() -> Result<Json<Value>, ApiError> {
    let device = koharu_runtime::device().await?;
    Ok(Json(json!({
        "device": device.description,
        "backend": format!("{:?}", device.backend),
        "memory": device.memory(),
    })))
}

async fn unload(State(app): State<Arc<App>>) -> &'static str {
    app.pipeline.unload();
    "ok"
}

/// The models each stage can use: Koharu's own and each connected vendor's
/// language models, and the vision models by id.
async fn models(State(app): State<Arc<App>>) -> Json<Value> {
    let translators = koharu_llm::models().await;
    let recognizers: Vec<Value> = Recognizer::VARIANTS
        .iter()
        .map(|model| json!({"id": model, "name": model.to_string()}))
        .collect();
    let inpainters: Vec<Value> = Inpainter::VARIANTS
        .iter()
        .map(|model| json!({"id": model, "name": model.to_string()}))
        .collect();
    Json(json!({
        "translators": translators,
        "recognizers": recognizers,
        "inpainters": inpainters,
        "current": current(&app.config.lock()),
        "agent_model": app.agent_model,
    }))
}

fn current(config: &PipelineConfig) -> Value {
    json!({
        "translator": config.translator.model,
        "recognizer": config.recognizer.model,
        "inpainter": config.inpainter.model,
    })
}

async fn set_config(
    State(app): State<Arc<App>>,
    Json(models): Json<Models>,
) -> Result<Json<Value>, ApiError> {
    let config = apply_models(&app, &models).map_err(|error| bad(error.to_string()))?;
    Ok(Json(current(&config)))
}

/// The settings with `models` applied, which every later job starts from. A
/// model switched away from leaves memory once the work queued with it ends.
fn apply_models(app: &App, models: &Models) -> Result<PipelineConfig> {
    let recognizer = models
        .recognizer
        .as_deref()
        .map(parse::<Recognizer>)
        .transpose()?;
    let inpainter = models
        .inpainter
        .as_deref()
        .map(parse::<Inpainter>)
        .transpose()?;
    let translator = models
        .translator
        .as_deref()
        .map(str::trim)
        .filter(|model| !model.is_empty())
        .map(str::to_owned);
    if let Some(model) = &translator
        && !model.contains('/')
    {
        anyhow::bail!("`{model}` names no model: expected `vendor/model`");
    }
    let mut config = app.config.lock();
    let before = config.clone();
    if let Some(model) = translator {
        config.translator.model = model;
    }
    if let Some(model) = recognizer {
        config.recognizer.model = model;
    }
    if let Some(model) = inpainter {
        config.inpainter.model = model;
    }
    if *config != before {
        app.pipeline.set_config(config.clone());
    }
    Ok(config.clone())
}

// --- Jobs ---

fn with_job(app: &App, id: u64, edit: impl FnOnce(&mut Job)) {
    if let Some(job) = app.jobs.lock().get_mut(&id) {
        edit(job);
    }
}

fn set(app: &App, id: u64, state: JobState) {
    with_job(app, id, |job| job.state = state);
}

async fn list_jobs(State(app): State<Arc<App>>) -> Json<Value> {
    let jobs = app.jobs.lock();
    let listed: Vec<Value> = jobs
        .iter()
        .rev()
        .map(|(id, job)| {
            json!({
                "id": id,
                "pages": job.pages,
                "stages": job.stages,
                "target": job.target.to_string(),
                "state": job.state.kind(),
            })
        })
        .collect();
    Json(Value::Array(listed))
}

async fn get_job(
    State(app): State<Arc<App>>,
    UrlPath(id): UrlPath<u64>,
) -> Result<Json<Value>, ApiError> {
    let jobs = app.jobs.lock();
    let job = jobs.get(&id).ok_or_else(|| not_found(id))?;
    Ok(Json(json!({
        "id": id,
        "pages": job.pages,
        "stages": job.stages,
        "target": job.target.to_string(),
        "instructions": job.instructions,
        "status": job.state,
    })))
}

/// Takes the upload, checks it, and starts the job; answers at once.
async fn create_job(
    State(app): State<Arc<App>>,
    upload: Multipart,
) -> Result<Json<Value>, ApiError> {
    let (options, pages) = read_upload(upload).await?;
    if pages.is_empty() {
        return Err(bad("no page images were uploaded"));
    }
    let id = app.next.fetch_add(1, Ordering::Relaxed) + 1;
    let dir = app.jobs_dir.join(id.to_string());
    // A previous run's job under this id would keep `Session::create` from
    // making the project file.
    if dir.exists() {
        fs_err::remove_dir_all(&dir).map_err(anyhow::Error::from)?;
    }
    fs_err::create_dir_all(&dir).map_err(anyhow::Error::from)?;
    let count = pages.len();
    app.jobs.lock().insert(
        id,
        Job {
            dir,
            pages: count,
            stages: options.stages.clone(),
            target: options.target.clone(),
            instructions: options.instructions.clone(),
            session: None,
            agent: None,
            busy: Arc::default(),
            cancelled: Arc::default(),
            state: JobState::Queued,
        },
    );
    for dir in prune(&app) {
        let _ = tokio::task::spawn_blocking(move || fs_err::remove_dir_all(dir));
    }
    tracing::info!(
        "job {id}: {count} pages, {:?} into {}",
        options.stages,
        options.target
    );
    let worker = Arc::clone(&app);
    tokio::spawn(async move {
        let state = match run_job(&worker, id, pages, options).await {
            Ok(state) => state,
            Err(error) => {
                tracing::error!("job {id} failed: {error:#}");
                JobState::Failed {
                    error: format!("{error:#}"),
                }
            }
        };
        set(&worker, id, state);
    });
    Ok(Json(json!({"id": id, "pages": count})))
}

/// Closes the oldest finished jobs past `max_jobs`, and answers their
/// folders for removal.
fn prune(app: &App) -> Vec<PathBuf> {
    let mut jobs = app.jobs.lock();
    let mut removed = Vec::new();
    while jobs.len() > app.max_jobs {
        let oldest = jobs
            .iter()
            .find(|(_, job)| job.state.settled() && job.busy.try_lock().is_ok())
            .map(|(id, _)| *id);
        let Some(id) = oldest else {
            break;
        };
        if let Some(job) = jobs.remove(&id) {
            removed.push(job.dir.clone());
            // The project file closes with its session, before its folder
            // goes.
            drop(job);
        }
    }
    removed
}

/// The upload's options and page images, each archive's pages in name order.
async fn read_upload(
    mut upload: Multipart,
) -> Result<(Options, Vec<(String, Vec<u8>)>), ApiError> {
    let mut target = "en".to_owned();
    let mut stages = String::new();
    let mut instructions = String::new();
    let mut models = Models::default();
    let mut files: Vec<(String, Vec<u8>)> = Vec::new();
    while let Some(field) = upload.next_field().await? {
        let name = field.name().unwrap_or_default().to_owned();
        let file_name = field.file_name().map(str::to_owned);
        let data = field.bytes().await?;
        let text = || String::from_utf8_lossy(&data).trim().to_owned();
        let model = || Some(text()).filter(|model| !model.is_empty());
        match name.as_str() {
            "target" => target = text(),
            "stages" => stages = text(),
            "instructions" => instructions = String::from_utf8_lossy(&data).into_owned(),
            "translator" => models.translator = model(),
            "recognizer" => models.recognizer = model(),
            "inpainter" => models.inpainter = model(),
            "pages" | "page" | "file" | "files" => {
                let name = file_name.unwrap_or_else(|| format!("page{}", files.len() + 1));
                files.push((name, data.to_vec()));
            }
            other => return Err(bad(format!("unknown field `{other}`"))),
        }
    }
    let options = Options {
        stages: parse_stages(&stages)?,
        target: parse_target(&target)?,
        instructions,
        models,
    };
    Ok((options, expand(files)?))
}

fn parse_target(tag: &str) -> Result<LanguageIdentifier, ApiError> {
    LanguageIdentifier::try_from_str(tag.trim())
        .map_err(|_| bad(format!("`{tag}` isn't a BCP 47 language tag")))
}

/// The stages `list` names, comma-separated; every stage for none.
fn parse_stages(list: &str) -> Result<BTreeSet<Stage>, ApiError> {
    if list.trim().is_empty() {
        return Ok(Stage::VARIANTS.iter().copied().collect());
    }
    list.split(',')
        .map(|stage| parse::<Stage>(stage.trim()).map_err(|error| bad(error.to_string())))
        .collect()
}

/// The page images among `files`, each archive's in name order, every one
/// checked to be an image Koharu reads.
fn expand(files: Vec<(String, Vec<u8>)>) -> Result<Vec<(String, Vec<u8>)>, ApiError> {
    let mut pages = Vec::new();
    for (name, bytes) in files {
        let lower = name.to_ascii_lowercase();
        if lower.ends_with(".zip") || lower.ends_with(".cbz") {
            pages.extend(unzip(&bytes).map_err(|error| bad(format!("{name}: {error:#}")))?);
            continue;
        }
        let readable = image::ImageReader::new(Cursor::new(&bytes))
            .with_guessed_format()
            .ok()
            .and_then(|reader| reader.into_dimensions().ok());
        if readable.is_none() {
            return Err(bad(format!("{name} isn't a PNG, JPEG, or WebP image")));
        }
        pages.push((name, bytes));
    }
    Ok(pages)
}

/// A ZIP or CBZ archive's images in name order, numbers by value, without
/// the files an OS leaves in an archive.
fn unzip(bytes: &[u8]) -> Result<Vec<(String, Vec<u8>)>> {
    let mut archive = zip::ZipArchive::new(Cursor::new(bytes))?;
    let mut images = Vec::new();
    for index in 0..archive.len() {
        let mut member = archive.by_index(index)?;
        if member.is_dir() {
            continue;
        }
        let name = member
            .name()
            .with_context(|| format!("member {index} has no readable name"))?
            .into_owned();
        let lower = name.to_ascii_lowercase();
        let image = [".png", ".jpg", ".jpeg", ".webp"]
            .iter()
            .any(|extension| lower.ends_with(extension));
        let metadata = name.contains("__MACOSX")
            || Path::new(&name)
                .file_name()
                .is_some_and(|file| file.to_string_lossy().starts_with("._"));
        if !image || metadata {
            continue;
        }
        let mut data = Vec::new();
        member.read_to_end(&mut data)?;
        images.push((name, data));
    }
    alphanumeric_sort::sort_slice_by_str_key(&mut images, |(name, _)| name);
    Ok(images)
}

/// Makes the job's project and runs its stages; answers how it ended.
async fn run_job(
    app: &App,
    id: u64,
    pages: Vec<(String, Vec<u8>)>,
    options: Options,
) -> Result<JobState> {
    let (dir, busy, cancelled) = {
        let jobs = app.jobs.lock();
        let job = jobs.get(&id).context("the job is gone")?;
        (
            job.dir.clone(),
            Arc::clone(&job.busy),
            Arc::clone(&job.cancelled),
        )
    };
    let _busy = busy.lock().await;
    let _turn = app.turn.acquire().await?;
    if cancelled.load(Ordering::Relaxed) {
        return Ok(JobState::Cancelled);
    }
    set(app, id, running(0, 0, None));

    // The project, as the app imports files: each image kept byte for byte
    // as a page's background.
    let path = dir.join("pages.khr");
    let session = tokio::task::spawn_blocking(move || -> Result<Session> {
        let mut session = Session::create(&path)?;
        let mut ops = Vec::with_capacity(pages.len() * 2);
        for (index, (_, bytes)) in pages.into_iter().enumerate() {
            let background = Some(Hash::of(&bytes));
            ops.push(Op::AddFile { bytes });
            ops.push(Op::AddPage {
                id: None,
                index,
                size: None,
                background,
            });
        }
        session.apply(Edit {
            label: "Import".to_owned(),
            ops,
        })?;
        Ok(session)
    })
    .await??;
    let session = Arc::new(Mutex::new(session));
    with_job(app, id, |job| job.session = Some(Arc::clone(&session)));
    process(app, id, &dir, &session, options).await
}

/// Runs `options` again on a job's open project; answers how it ended.
async fn again(app: &App, id: u64, options: Options) -> Result<JobState> {
    let (dir, busy, cancelled, session) = {
        let jobs = app.jobs.lock();
        let job = jobs.get(&id).context("the job is gone")?;
        (
            job.dir.clone(),
            Arc::clone(&job.busy),
            Arc::clone(&job.cancelled),
            Arc::clone(job.session.as_ref().context("the job has no project")?),
        )
    };
    let _busy = busy.lock().await;
    let _turn = app.turn.acquire().await?;
    if cancelled.load(Ordering::Relaxed) {
        return Ok(JobState::Cancelled);
    }
    set(app, id, running(0, 0, None));
    process(app, id, &dir, &session, options).await
}

/// Runs the stages on the project and renders what they left; the caller
/// holds the job and the turn.
async fn process(
    app: &App,
    id: u64,
    dir: &Path,
    session: &Arc<Mutex<Session>>,
    options: Options,
) -> Result<JobState> {
    let mut config = apply_models(app, &options.models)?;
    config.translator.target = options.target;
    config.translator.instructions = options.instructions;
    let project = session.lock().id();
    let run = app.pipeline.run(
        Arc::clone(session),
        Request {
            target: Target::Project,
            stages: options.stages,
            config: Some(config),
        },
    )?;

    let outcome = loop {
        let events = app.pipeline.history(run, Duration::from_secs(5)).await?;
        let steps = events
            .iter()
            .find_map(|event| match event {
                Event::Queued { steps, .. } => Some(*steps),
                _ => None,
            })
            .unwrap_or(0);
        let finished = events
            .iter()
            .filter(|event| matches!(event, Event::Finished { .. }))
            .count();
        let current = events.iter().rev().find_map(|event| match event {
            Event::Started { page, stage, .. } => {
                Some(format!("{stage} on page {}", number(session, *page)))
            }
            _ => None,
        });
        set(app, id, running(finished, steps, current));
        if let Some(Event::Ended { outcome, .. }) = events.last() {
            break outcome.clone();
        }
    };

    let state = match outcome {
        Outcome::Done => publish(app, id, session, dir)
            .await
            .map(|pages| JobState::Done { pages }),
        Outcome::Failed {
            page,
            stage,
            reason,
        } => Ok(JobState::Failed {
            error: format!(
                "{stage} failed on page {}: {reason}",
                number(session, page)
            ),
        }),
        Outcome::Stopped => Ok(JobState::Cancelled),
    };
    // The translation conversation of a run that's over.
    app.pipeline.close(project);
    state
}

/// The page's number in the project, from 1.
fn number(session: &Mutex<Session>, page: PageId) -> usize {
    session
        .lock()
        .project()
        .pages()
        .position(|found| found.id == page)
        .map_or(0, |at| at + 1)
}

/// Renders every page as the app exports it, and without its lettering, to
/// `dir`, and reads each page's layout and texts.
async fn publish(
    app: &App,
    id: u64,
    session: &Arc<Mutex<Session>>,
    dir: &Path,
) -> Result<Vec<PageInfo>> {
    set(app, id, JobState::Rendering);
    let renderer = app.renderer.clone();
    let session = Arc::clone(session);
    let dir = dir.to_owned();
    tokio::task::spawn_blocking(move || -> Result<Vec<PageInfo>> {
        let session = session.lock();
        let project = session.project();
        let mut pages = Vec::new();
        for (index, page) in project.pages().enumerate() {
            let number = index + 1;
            let raster = renderer.render(&session, page.id)?.rasterize()?;
            fs_err::write(dir.join(format!("{number:03}.png")), png(&raster.image)?)?;
            let cleaned = cleaned(&session, &project, page)?.unwrap_or(raster.image);
            fs_err::write(dir.join(format!("{number:03}.cleaned.png")), png(&cleaned)?)?;
            pages.push(describe(&project, page, number)?);
        }
        Ok(pages)
    })
    .await?
}

fn png(image: &RgbaImage) -> Result<Vec<u8>> {
    let mut out = Cursor::new(Vec::new());
    image.write_to(&mut out, image::ImageFormat::Png)?;
    Ok(out.into_inner())
}

/// The page's layout and texts in reading order.
fn describe(project: &Project, page: &Page, number: usize) -> Result<PageInfo> {
    let bbox = |rect: kurbo::Rect| [rect.x0, rect.y0, rect.x1, rect.y1];
    let mut info = PageInfo {
        page: number,
        width: page.size.width,
        height: page.size.height,
        panels: Vec::new(),
        balloons: Vec::new(),
        blocks: Vec::new(),
    };
    for layer in project.layers_of(page.id)? {
        match &layer.content {
            Content::Panel(panel) => info
                .panels
                .push(bbox(kurbo::BezPath::from(&panel.outline).bounding_box())),
            Content::Balloon(balloon) => info
                .balloons
                .push(bbox(kurbo::BezPath::from(&balloon.outline).bounding_box())),
            Content::Block(block) => {
                let in_balloon = matches!(
                    project.parent(layer.id)?.1,
                    Some(Layer {
                        content: Content::Balloon(_),
                        ..
                    })
                );
                info.blocks.push(BlockInfo {
                    role: match block.role {
                        Role::Text => "text",
                        Role::Sfx => "sfx",
                    },
                    bbox: block.source.bounds().map(bbox),
                    in_balloon,
                    visible: layer.visible,
                    source: block.source.content.text(),
                    translation: block
                        .translation
                        .as_ref()
                        .map(|translation| translation.content.text()),
                });
            }
            _ => {}
        }
    }
    Ok(info)
}

/// The page's scan with each shown block's clean-up laid over it through the
/// block's mask, and no lettering: what a cleaner hands a typesetter. `None`
/// for a page with no scan.
fn cleaned(session: &Session, project: &Project, page: &Page) -> Result<Option<RgbaImage>> {
    let Some(background) = page.background else {
        return Ok(None);
    };
    let mut image = image::load_from_memory(&session.file(&background.file)?)?.to_rgba8();
    let (width, height) = image.dimensions();
    for layer in project.layers_of(page.id)? {
        let Content::Block(block) = &layer.content else {
            continue;
        };
        if !layer.visible || !block.cleanup.visible {
            continue;
        }
        // A shown mask with nothing in it shows none of the clean-up; a
        // hidden mask shows all of it, as in the app.
        let mask = match block.mask.visible {
            true => match session.tiles(block.mask.id, None)?.coverage(width, height) {
                Some(mask) => Some(mask),
                None => continue,
            },
            false => None,
        };
        let shown = |x: u32, y: u32| match &mask {
            Some((rect, mask)) => {
                let (dx, dy) = (
                    i64::from(x) - i64::from(rect.x),
                    i64::from(y) - i64::from(rect.y),
                );
                dx >= 0
                    && dy >= 0
                    && dx < i64::from(rect.width)
                    && dy < i64::from(rect.height)
                    && mask.get_pixel(dx as u32, dy as u32).0[0] != 0
            }
            None => true,
        };
        let tiles = session.tiles(block.cleanup.id, None)?;
        let size = i64::from(Tile::SIZE);
        for tile in &tiles.tiles {
            let left = i64::from(tile.x) * size + i64::from(tiles.offset.x);
            let top = i64::from(tile.y) * size + i64::from(tiles.offset.y);
            for (at, pixel) in tile.pixels.chunks_exact(4).enumerate() {
                let (x, y) = (left + at as i64 % size, top + at as i64 / size);
                if pixel[3] == 0
                    || x < 0
                    || y < 0
                    || x >= i64::from(width)
                    || y >= i64::from(height)
                {
                    continue;
                }
                let (x, y) = (x as u32, y as u32);
                if shown(x, y) {
                    image.put_pixel(x, y, Rgba([pixel[0], pixel[1], pixel[2], 255]));
                }
            }
        }
    }
    Ok(Some(image))
}

async fn rerun_job(
    State(app): State<Arc<App>>,
    UrlPath(id): UrlPath<u64>,
    Json(request): Json<RerunRequest>,
) -> Result<Json<Value>, ApiError> {
    let stages = match &request.stages {
        Some(list) => Some(parse_stages(list)?),
        None => None,
    };
    let target = request.target.as_deref().map(parse_target).transpose()?;
    let options = {
        let mut jobs = app.jobs.lock();
        let job = jobs.get_mut(&id).ok_or_else(|| not_found(id))?;
        if !job.state.settled() {
            return Err(conflict(format!("job {id} is still running")));
        }
        if job.session.is_none() {
            return Err(bad(format!("job {id} has no project to run again")));
        }
        let options = Options {
            stages: stages.unwrap_or_else(|| job.stages.clone()),
            target: target.unwrap_or_else(|| job.target.clone()),
            instructions: request
                .instructions
                .clone()
                .unwrap_or_else(|| job.instructions.clone()),
            models: Models {
                translator: request.translator.clone(),
                recognizer: request.recognizer.clone(),
                inpainter: request.inpainter.clone(),
            },
        };
        job.stages = options.stages.clone();
        job.target = options.target.clone();
        job.instructions = options.instructions.clone();
        job.cancelled.store(false, Ordering::Relaxed);
        job.state = JobState::Queued;
        options
    };
    tracing::info!(
        "job {id}: again, {:?} into {}",
        options.stages,
        options.target
    );
    let worker = Arc::clone(&app);
    tokio::spawn(async move {
        let state = match again(&worker, id, options).await {
            Ok(state) => state,
            Err(error) => {
                tracing::error!("job {id} failed: {error:#}");
                JobState::Failed {
                    error: format!("{error:#}"),
                }
            }
        };
        set(&worker, id, state);
    });
    Ok(Json(json!({"id": id})))
}

/// Stops a queued job before it starts, or a running one at its next model
/// call; what it finished stays.
async fn cancel_job(
    State(app): State<Arc<App>>,
    UrlPath(id): UrlPath<u64>,
) -> Result<&'static str, ApiError> {
    let running = {
        let mut jobs = app.jobs.lock();
        let job = jobs.get_mut(&id).ok_or_else(|| not_found(id))?;
        match job.state.kind() {
            "queued" => {
                job.cancelled.store(true, Ordering::Relaxed);
                job.state = JobState::Cancelled;
                false
            }
            "running" => {
                job.cancelled.store(true, Ordering::Relaxed);
                true
            }
            _ => return Err(conflict(format!("job {id} isn't queued or running"))),
        }
    };
    if running {
        // One run at a time, so this is the job's.
        app.pipeline.stop();
    }
    Ok("cancelling")
}

async fn delete_job(
    State(app): State<Arc<App>>,
    UrlPath(id): UrlPath<u64>,
) -> Result<&'static str, ApiError> {
    let job = {
        let mut jobs = app.jobs.lock();
        let job = jobs.get(&id).ok_or_else(|| not_found(id))?;
        if !job.state.settled() || job.busy.try_lock().is_err() {
            return Err(conflict(format!("job {id} is busy")));
        }
        jobs.remove(&id)
    };
    if let Some(job) = job {
        let dir = job.dir.clone();
        // The project file closes with its session, before its folder goes.
        drop(job);
        let _ = tokio::task::spawn_blocking(move || fs_err::remove_dir_all(dir)).await;
    }
    Ok("deleted")
}

// --- The agent ---

/// A turn of the agent's conversation about the job's project, on the
/// worker's agent model; the pages render again after it, since the agent
/// may have changed them.
async fn ask(
    State(app): State<Arc<App>>,
    UrlPath(id): UrlPath<u64>,
    Json(request): Json<AskRequest>,
) -> Result<Json<Value>, ApiError> {
    let reference = app
        .agent_model
        .clone()
        .ok_or_else(|| bad("this worker has no agent model: start it with --agent-model"))?;
    if request.prompt.trim().is_empty() {
        return Err(bad("the prompt is empty"));
    }
    let (dir, busy, session, agent) = {
        let jobs = app.jobs.lock();
        let job = jobs.get(&id).ok_or_else(|| not_found(id))?;
        if !job.state.settled() {
            return Err(conflict(format!("job {id} is still running")));
        }
        let session = job
            .session
            .clone()
            .ok_or_else(|| bad(format!("job {id} has no project")))?;
        (
            job.dir.clone(),
            Arc::clone(&job.busy),
            session,
            job.agent.clone(),
        )
    };
    let _busy = busy.lock().await;
    let model = agent_model(&app, &reference).await?;
    let agent = match agent {
        Some(agent) => agent,
        None => {
            let tools = Tools::new(
                Arc::clone(&session),
                app.pipeline.clone(),
                app.renderer.clone(),
            );
            let agent = Arc::new(tokio::sync::Mutex::new(Agent::new(
                tools,
                model,
                LlmContext::new(SummaryPrompts::DEFAULT),
            )));
            with_job(&app, id, |job| job.agent = Some(Arc::clone(&agent)));
            agent
        }
    };
    let config: AgentConfig =
        serde_json::from_value(json!({"model": reference, "reasoning": request.reasoning}))
            .map_err(|error| bad(format!("reasoning: {error}")))?;
    let calls = Arc::new(Mutex::new(Vec::<Value>::new()));
    let answer = {
        let heard = Arc::clone(&calls);
        let mut agent = agent.lock().await;
        agent.set_config(config);
        agent
            .run(&request.prompt, move |event| {
                if !matches!(event, AgentEvent::Text { .. } | AgentEvent::Reasoning { .. }) {
                    heard
                        .lock()
                        .push(serde_json::to_value(&event).unwrap_or(Value::Null));
                }
            })
            .await?
    };
    // What the agent changed shows in the pages again.
    let pages = publish(&app, id, &session, &dir).await?;
    set(&app, id, JobState::Done { pages });
    let calls = calls.lock().clone();
    Ok(Json(json!({"answer": answer, "calls": calls})))
}

/// The agent's model, connected the first time and kept.
async fn agent_model(app: &App, reference: &str) -> Result<DynModel<Completion>> {
    let mut cached = app.agent.lock().await;
    if let Some(model) = cached.as_ref() {
        return Ok(model.clone());
    }
    let model = koharu_llm::connect(reference, Size::AGENT)
        .await
        .with_context(|| format!("couldn't connect the agent's model {reference}"))?;
    *cached = Some(model.clone());
    Ok(model)
}

// --- Exports ---

/// A done job's handles: its folder, its lock, its session, and its page
/// count; `page` checked against the count when given.
fn done_job(
    app: &App,
    id: u64,
    page: Option<usize>,
) -> Result<(PathBuf, Arc<tokio::sync::Mutex<()>>, Arc<Mutex<Session>>, usize), ApiError> {
    let jobs = app.jobs.lock();
    let job = jobs.get(&id).ok_or_else(|| not_found(id))?;
    if !matches!(job.state, JobState::Done { .. }) {
        return Err(conflict(format!("job {id} isn't done")));
    }
    if let Some(page) = page
        && (page == 0 || page > job.pages)
    {
        return Err(ApiError(
            StatusCode::NOT_FOUND,
            format!("job {id} has pages 1 to {}", job.pages),
        ));
    }
    let session = job
        .session
        .clone()
        .ok_or_else(|| bad(format!("job {id} has no project")))?;
    Ok((job.dir.clone(), Arc::clone(&job.busy), session, job.pages))
}

fn attachment(bytes: Vec<u8>, mime: &'static str, name: &str) -> Response {
    (
        [
            (header::CONTENT_TYPE, mime.to_owned()),
            (
                header::CONTENT_DISPOSITION,
                format!("attachment; filename=\"{name}\""),
            ),
        ],
        bytes,
    )
        .into_response()
}

async fn page_file(app: &App, id: u64, page: usize, suffix: &str) -> Result<Response, ApiError> {
    let (dir, ..) = done_job(app, id, Some(page))?;
    let path = dir.join(format!("{page:03}{suffix}.png"));
    let bytes = tokio::fs::read(&path)
        .await
        .with_context(|| format!("couldn't read {}", path.display()))?;
    Ok(([(header::CONTENT_TYPE, "image/png")], bytes).into_response())
}

async fn get_page(
    State(app): State<Arc<App>>,
    UrlPath((id, page)): UrlPath<(u64, usize)>,
) -> Result<Response, ApiError> {
    page_file(&app, id, page, "").await
}

async fn get_cleaned(
    State(app): State<Arc<App>>,
    UrlPath((id, page)): UrlPath<(u64, usize)>,
) -> Result<Response, ApiError> {
    page_file(&app, id, page, ".cleaned").await
}

/// The names PSD layers take where the scene has none.
fn labels() -> Labels {
    Labels {
        background: "Background".to_owned(),
        cleanup: "Cleanup".to_owned(),
        outline: "Outline".to_owned(),
    }
}

/// Page `page` of the open project as a layered PSD.
fn psd(session: &Session, renderer: &Renderer, page: usize) -> Result<Vec<u8>> {
    let id = session
        .project()
        .pages()
        .nth(page - 1)
        .map(|found| found.id)
        .with_context(|| format!("the project has no page {page}"))?;
    Ok(koharu_psd::export(session, renderer, id, &labels())?)
}

async fn get_psd(
    State(app): State<Arc<App>>,
    UrlPath((id, page)): UrlPath<(u64, usize)>,
) -> Result<Response, ApiError> {
    let (_, busy, session, _) = done_job(&app, id, Some(page))?;
    let _busy = busy.lock().await;
    let renderer = app.renderer.clone();
    let bytes = tokio::task::spawn_blocking(move || psd(&session.lock(), &renderer, page))
        .await
        .map_err(anyhow::Error::from)??;
    Ok(attachment(
        bytes,
        "image/vnd.adobe.photoshop",
        &format!("{page:03}.psd"),
    ))
}

async fn get_psd_zip(
    State(app): State<Arc<App>>,
    UrlPath(id): UrlPath<u64>,
) -> Result<Response, ApiError> {
    let (_, busy, session, pages) = done_job(&app, id, None)?;
    let _busy = busy.lock().await;
    let renderer = app.renderer.clone();
    let bytes = tokio::task::spawn_blocking(move || -> Result<Vec<u8>> {
        let session = session.lock();
        let mut archive = zip::ZipWriter::new(Cursor::new(Vec::new()));
        for page in 1..=pages {
            // A PSD's uncompressed channels deflate well.
            let options = zip::write::SimpleFileOptions::default()
                .compression_method(zip::CompressionMethod::Deflated);
            archive.start_file(format!("{page:03}.psd"), options)?;
            archive.write_all(&psd(&session, &renderer, page)?)?;
        }
        Ok(archive.finish()?.into_inner())
    })
    .await
    .map_err(anyhow::Error::from)??;
    Ok(attachment(bytes, "application/zip", "pages-psd.zip"))
}

async fn get_cbz(
    State(app): State<Arc<App>>,
    UrlPath(id): UrlPath<u64>,
) -> Result<Response, ApiError> {
    let (dir, _, _, pages) = done_job(&app, id, None)?;
    let bytes = tokio::task::spawn_blocking(move || -> Result<Vec<u8>> {
        let mut archive = zip::ZipWriter::new(Cursor::new(Vec::new()));
        for page in 1..=pages {
            // Already-compressed PNGs gain nothing from deflate.
            let options = zip::write::SimpleFileOptions::default()
                .compression_method(zip::CompressionMethod::Stored);
            archive.start_file(format!("{page:03}.png"), options)?;
            archive.write_all(&fs_err::read(dir.join(format!("{page:03}.png")))?)?;
        }
        Ok(archive.finish()?.into_inner())
    })
    .await
    .map_err(anyhow::Error::from)??;
    Ok(attachment(bytes, "application/vnd.comicbook+zip", "pages.cbz"))
}
