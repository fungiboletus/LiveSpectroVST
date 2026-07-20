#![forbid(unsafe_code)]

use egui::{Color32, ColorImage, TextureHandle, TextureOptions, Vec2};
use nice_plug::prelude::*;
use nice_plug_egui::{EguiState, create_egui_editor};
use realfft::num_complex::Complex32;
use realfft::{RealFftPlanner, RealToComplex};
use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock, Weak};
use std::time::Instant;

const FFT_SIZES: [usize; 6] = [256, 512, 1024, 2048, 4096, 8192];
const SCROLL_BEATS_PER_PIXEL: [f32; 7] = [0.25, 0.5, 1.0, 2.0, 4.0, 0.125, 0.0625];
const SCROLL_LABELS: [&str; 7] = ["1/16", "1/8", "1/4", "1/2", "1 bar", "1/32", "1/64"];
const SCROLL_DISPLAY_ORDER: [usize; 7] = [6, 5, 0, 1, 2, 3, 4];
const MAX_FFT_SIZE: usize = 8192;
const SPECTRUM_BINS: usize = 192;
const HISTORY_COLUMNS: usize = 600;
const WINDOW_WIDTH: u32 = 760;
const WINDOW_HEIGHT: u32 = 480;
const SOURCE_PANEL_WIDTH: f32 = 138.0;
const SOURCE_PANEL_PADDING: f32 = 10.0;
const FREQUENCY_SCALE_WIDTH: f32 = 40.0;
const SOURCE_ACTIVITY_TIMEOUT_MS: u64 = 1_500;

pub struct LiveSpectroVst {
    params: Arc<LiveSpectroParams>,
    sample_rate: f32,
    samples_since_frame: usize,
    rolling_samples: [f32; MAX_FFT_SIZE],
    rolling_position: usize,
    spectrum_sequence: u64,
    fft_plans: Vec<Arc<dyn RealToComplex<f32>>>,
    fft_input: Vec<f32>,
    fft_output: Vec<Complex32>,
    fft_scratch: Vec<Complex32>,
    source: Arc<SharedSource>,
    initial_gui_state: Option<GuiState>,
}

#[derive(Params)]
struct LiveSpectroParams {
    #[persist = "editor-state"]
    editor_state: Arc<EguiState>,

    #[id = "fft-size"]
    fft_size: IntParam,

    #[id = "scroll-division"]
    scroll_division: IntParam,
}

struct SpectrumFrame {
    magnitudes: [f32; SPECTRUM_BINS],
    sequence: u64,
    max_frequency: f32,
    tempo: f32,
}

struct SharedSource {
    id: u64,
    name: Mutex<Option<String>>,
    name_generation: AtomicU64,
    magnitudes: [AtomicU32; SPECTRUM_BINS],
    sequence: AtomicU64,
    max_frequency: AtomicU32,
    tempo: AtomicU32,
    last_process_ms: AtomicU64,
}

#[derive(Clone)]
struct HistorySource {
    id: u64,
    magnitudes: [f32; SPECTRUM_BINS],
}

#[derive(Clone)]
struct HistoryColumn {
    sources: Vec<HistorySource>,
}

struct SharedHistory {
    columns: VecDeque<HistoryColumn>,
    generation: u64,
    column_accumulator: f32,
    last_update_ms: u64,
}

static NEXT_SOURCE_ID: AtomicU64 = AtomicU64::new(1);
static SOURCES: OnceLock<Mutex<Vec<Weak<SharedSource>>>> = OnceLock::new();
static SHARED_HISTORY: OnceLock<Mutex<SharedHistory>> = OnceLock::new();
static UI_CLOCK: OnceLock<Instant> = OnceLock::new();
static UI_HEARTBEAT_MS: AtomicU64 = AtomicU64::new(0);
static GLOBAL_FFT_SIZE: AtomicU32 = AtomicU32::new(5);
static GLOBAL_SCROLL_DIVISION: AtomicU32 = AtomicU32::new(5);
static GLOBAL_WINDOW_SIZE: AtomicU64 =
    AtomicU64::new(pack_window_size(WINDOW_WIDTH, WINDOW_HEIGHT));

struct GuiState {
    history: ColorImage,
    texture: Option<TextureHandle>,
    enabled_sources: HashSet<u64>,
    known_sources: HashSet<u64>,
    rendered_sources: HashSet<u64>,
    source_names: HashMap<u64, (u64, String)>,
    history_generation: u64,
    column_offset: f32,
    last_window_size: (u32, u32),
    pending_window_size: Option<(u32, u32)>,
}

const fn pack_window_size(width: u32, height: u32) -> u64 {
    ((width as u64) << 32) | height as u64
}

fn unpack_window_size(size: u64) -> (u32, u32) {
    ((size >> 32) as u32, size as u32)
}

impl SharedSource {
    fn new() -> Arc<Self> {
        let source = Arc::new(Self {
            id: NEXT_SOURCE_ID.fetch_add(1, Ordering::Relaxed),
            name: Mutex::new(None),
            name_generation: AtomicU64::new(0),
            magnitudes: std::array::from_fn(|_| AtomicU32::new(0.0_f32.to_bits())),
            sequence: AtomicU64::new(0),
            max_frequency: AtomicU32::new(22_050.0_f32.to_bits()),
            tempo: AtomicU32::new(120.0_f32.to_bits()),
            last_process_ms: AtomicU64::new(ui_clock_ms()),
        });
        let registry = SOURCES.get_or_init(|| Mutex::new(Vec::new()));
        registry
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .push(Arc::downgrade(&source));
        source
    }

    fn publish(&self, frame: &SpectrumFrame) {
        let version = frame.sequence.wrapping_mul(2);
        self.sequence
            .store(version.wrapping_sub(1), Ordering::Release);
        for (output, magnitude) in self.magnitudes.iter().zip(frame.magnitudes) {
            output.store(magnitude.to_bits(), Ordering::Relaxed);
        }
        self.max_frequency
            .store(frame.max_frequency.to_bits(), Ordering::Relaxed);
        self.tempo.store(frame.tempo.to_bits(), Ordering::Relaxed);
        self.sequence.store(version, Ordering::Release);
    }

    fn set_name(&self, name: Option<&str>) {
        let name = name
            .map(str::trim)
            .filter(|name| !name.is_empty())
            .map(str::to_owned);
        let mut current_name = self
            .name
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if *current_name != name {
            *current_name = name;
            self.name_generation.fetch_add(1, Ordering::Release);
        }
    }

    fn display_name(&self) -> (u64, String) {
        let generation = self.name_generation.load(Ordering::Acquire);
        let name = self
            .name
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .clone()
            .unwrap_or_else(|| format!("Source {}", self.id));
        (generation, name)
    }

    fn snapshot(&self) -> SpectrumFrame {
        loop {
            let version = self.sequence.load(Ordering::Acquire);
            if !version.is_multiple_of(2) {
                std::hint::spin_loop();
                continue;
            }
            let mut frame = SpectrumFrame {
                magnitudes: std::array::from_fn(|index| {
                    f32::from_bits(self.magnitudes[index].load(Ordering::Relaxed))
                }),
                sequence: version / 2,
                max_frequency: f32::from_bits(self.max_frequency.load(Ordering::Relaxed)),
                tempo: f32::from_bits(self.tempo.load(Ordering::Relaxed)),
            };
            if version == self.sequence.load(Ordering::Acquire) {
                frame.sequence = version / 2;
                return frame;
            }
        }
    }
}

fn shared_sources() -> Vec<Arc<SharedSource>> {
    let registry = SOURCES.get_or_init(|| Mutex::new(Vec::new()));
    let mut sources = registry
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let now = ui_clock_ms();
    let live: Vec<_> = sources
        .iter()
        .filter_map(Weak::upgrade)
        .filter(|source| {
            now.saturating_sub(source.last_process_ms.load(Ordering::Relaxed))
                < SOURCE_ACTIVITY_TIMEOUT_MS
        })
        .collect();
    sources.retain(|source| source.strong_count() > 0);
    live
}

fn ui_clock_ms() -> u64 {
    UI_CLOCK.get_or_init(Instant::now).elapsed().as_millis() as u64 + 1
}

fn shared_view_active() -> bool {
    ui_clock_ms().saturating_sub(UI_HEARTBEAT_MS.load(Ordering::Relaxed)) < 500
}

fn update_shared_history(
    frames: &[(u64, SpectrumFrame)],
    seconds_per_pixel: f32,
    reset: bool,
) -> (u64, f32) {
    let history = SHARED_HISTORY.get_or_init(|| {
        Mutex::new(SharedHistory {
            columns: VecDeque::with_capacity(HISTORY_COLUMNS),
            generation: 0,
            column_accumulator: 0.0,
            last_update_ms: ui_clock_ms(),
        })
    });
    let mut history = history
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    if reset {
        history.columns.clear();
        history.column_accumulator = 0.0;
        history.generation = history.generation.wrapping_add(1);
        history.last_update_ms = ui_clock_ms();
    }

    let now = ui_clock_ms();
    let elapsed = now.saturating_sub(history.last_update_ms).min(100) as f32 / 1_000.0;
    history.last_update_ms = now;
    history.column_accumulator += elapsed / seconds_per_pixel;
    let new_columns = history.column_accumulator.floor() as usize;
    history.column_accumulator -= new_columns as f32;
    if new_columns > 0 {
        let column = HistoryColumn {
            sources: frames
                .iter()
                .map(|(id, frame)| HistorySource {
                    id: *id,
                    magnitudes: frame.magnitudes,
                })
                .collect(),
        };
        for _ in 0..new_columns.min(HISTORY_COLUMNS) {
            if history.columns.len() == HISTORY_COLUMNS {
                history.columns.pop_front();
            }
            history.columns.push_back(column.clone());
        }
        history.generation = history.generation.wrapping_add(1);
    }

    (history.generation, history.column_accumulator)
}

fn shared_history_snapshot() -> Vec<HistoryColumn> {
    SHARED_HISTORY
        .get_or_init(|| {
            Mutex::new(SharedHistory {
                columns: VecDeque::with_capacity(HISTORY_COLUMNS),
                generation: 0,
                column_accumulator: 0.0,
                last_update_ms: ui_clock_ms(),
            })
        })
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
        .columns
        .iter()
        .cloned()
        .collect()
}

impl Default for SpectrumFrame {
    fn default() -> Self {
        Self {
            magnitudes: [0.0; SPECTRUM_BINS],
            sequence: 0,
            max_frequency: 22_050.0,
            tempo: 120.0,
        }
    }
}

impl Default for LiveSpectroParams {
    fn default() -> Self {
        Self {
            editor_state: EguiState::from_size(WINDOW_WIDTH, WINDOW_HEIGHT),
            fft_size: IntParam::new("FFT Size", 5, IntRange::Linear { min: 0, max: 5 })
                .with_value_to_string(Arc::new(|index| {
                    FFT_SIZES[index.clamp(0, 5) as usize].to_string()
                }))
                .non_automatable(),
            scroll_division: IntParam::new(
                "Scroll Division",
                5,
                IntRange::Linear { min: 0, max: 6 },
            )
            .with_value_to_string(Arc::new(|index| {
                SCROLL_LABELS[index.clamp(0, 6) as usize].to_string()
            })),
        }
    }
}

impl Default for LiveSpectroVst {
    fn default() -> Self {
        let mut planner = RealFftPlanner::<f32>::new();
        let fft_plans: Vec<_> = FFT_SIZES
            .iter()
            .map(|size| planner.plan_fft_forward(*size))
            .collect();
        let max_plan = fft_plans.last().unwrap();
        let source = SharedSource::new();

        Self {
            params: Arc::new(LiveSpectroParams::default()),
            sample_rate: 44_100.0,
            samples_since_frame: 0,
            rolling_samples: [0.0; MAX_FFT_SIZE],
            rolling_position: 0,
            spectrum_sequence: 0,
            fft_input: max_plan.make_input_vec(),
            fft_output: max_plan.make_output_vec(),
            fft_scratch: max_plan.make_scratch_vec(),
            fft_plans,
            source,
            initial_gui_state: Some(GuiState {
                history: ColorImage::filled(
                    [HISTORY_COLUMNS, SPECTRUM_BINS],
                    Color32::from_rgb(5, 8, 14),
                ),
                texture: None,
                enabled_sources: HashSet::new(),
                known_sources: HashSet::new(),
                rendered_sources: HashSet::new(),
                source_names: HashMap::new(),
                history_generation: u64::MAX,
                column_offset: 0.0,
                last_window_size: (WINDOW_WIDTH, WINDOW_HEIGHT),
                pending_window_size: None,
            }),
        }
    }
}

impl LiveSpectroVst {
    fn analyze(&mut self, tempo: f32) {
        let plan_index = GLOBAL_FFT_SIZE.load(Ordering::Relaxed).clamp(0, 5) as usize;
        let fft_size = FFT_SIZES[plan_index];
        let plan = &self.fft_plans[plan_index];

        for index in 0..fft_size {
            let source = (self.rolling_position + MAX_FFT_SIZE - fft_size + index) % MAX_FFT_SIZE;
            let window =
                0.5 - 0.5 * (std::f32::consts::TAU * index as f32 / (fft_size - 1) as f32).cos();
            self.fft_input[index] = self.rolling_samples[source] * window;
        }

        plan.process_with_scratch(
            &mut self.fft_input[..fft_size],
            &mut self.fft_output[..fft_size / 2 + 1],
            &mut self.fft_scratch[..plan.get_scratch_len()],
        )
        .expect("FFT buffers must match the precomputed plan");

        let normalization = 2.0 / fft_size as f32;
        let min_frequency = 20.0_f32;
        let max_frequency = self.sample_rate * 0.5;
        let mut frame = SpectrumFrame::default();
        for (display_bin, magnitude) in frame.magnitudes.iter_mut().enumerate() {
            let fraction = display_bin as f32 / (SPECTRUM_BINS - 1) as f32;
            let frequency = min_frequency * (max_frequency / min_frequency).powf(fraction);
            let fft_bin = (frequency * fft_size as f32 / self.sample_rate)
                .round()
                .clamp(0.0, (fft_size / 2) as f32) as usize;
            let gain = self.fft_output[fft_bin].norm() * normalization;
            let db = util::gain_to_db(gain.max(1.0e-8));
            *magnitude = ((db + 90.0) / 90.0).clamp(0.0, 1.0);
        }
        self.spectrum_sequence = self.spectrum_sequence.wrapping_add(1);
        frame.sequence = self.spectrum_sequence;
        frame.max_frequency = max_frequency;
        frame.tempo = tempo;
        self.source.publish(&frame);
    }
}

impl Plugin for LiveSpectroVst {
    const NAME: &'static str = "Live Spectro";
    const VENDOR: &'static str = "Antoine";
    const URL: &'static str = "https://github.com/antoinep/LiveSpectroVST";
    const EMAIL: &'static str = "";
    const VERSION: &'static str = env!("CARGO_PKG_VERSION");

    const AUDIO_IO_LAYOUTS: &'static [AudioIOLayout] = &[
        AudioIOLayout {
            main_input_channels: NonZeroU32::new(2),
            main_output_channels: NonZeroU32::new(2),
            ..AudioIOLayout::const_default()
        },
        AudioIOLayout {
            main_input_channels: NonZeroU32::new(1),
            main_output_channels: NonZeroU32::new(1),
            ..AudioIOLayout::const_default()
        },
    ];

    type SysExMessage = ();
    type BackgroundTask = ();

    fn params(&self) -> Arc<dyn Params> {
        self.params.clone()
    }

    fn editor(&mut self, _async_executor: AsyncExecutor<Self>) -> Option<Box<dyn Editor>> {
        let params = self.params.clone();
        let editor_state = params.editor_state.clone();
        editor_state.set_requested_size(unpack_window_size(
            GLOBAL_WINDOW_SIZE.load(Ordering::Relaxed),
        ));
        let gui_state = self.initial_gui_state.take().unwrap();

        create_egui_editor(
            editor_state,
            gui_state,
            Default::default(),
            |_ctx, _queue, state| {
                // Texture IDs belong to one renderer and become invalid when a host closes the UI.
                state.texture = None;
            },
            move |ui, setter, _queue, state| {
                let current_window_size = params.editor_state.size();
                if state.pending_window_size == Some(current_window_size) {
                    state.pending_window_size = None;
                } else if current_window_size != state.last_window_size {
                    GLOBAL_WINDOW_SIZE.store(
                        pack_window_size(current_window_size.0, current_window_size.1),
                        Ordering::Relaxed,
                    );
                }
                let global_window_size =
                    unpack_window_size(GLOBAL_WINDOW_SIZE.load(Ordering::Relaxed));
                if current_window_size != global_window_size
                    && state.pending_window_size != Some(global_window_size)
                {
                    params.editor_state.set_requested_size(global_window_size);
                    state.pending_window_size = Some(global_window_size);
                }
                state.last_window_size = current_window_size;

                let now = ui_clock_ms();
                let start_new_session =
                    now.saturating_sub(UI_HEARTBEAT_MS.swap(now, Ordering::Relaxed)) >= 500;
                let sources = shared_sources();
                let live_source_ids: HashSet<_> = sources.iter().map(|source| source.id).collect();
                state
                    .enabled_sources
                    .retain(|id| live_source_ids.contains(id));
                state
                    .known_sources
                    .retain(|id| live_source_ids.contains(id));
                state
                    .source_names
                    .retain(|id, _| live_source_ids.contains(id));
                for source in &sources {
                    if state.known_sources.insert(source.id) {
                        state.enabled_sources.insert(source.id);
                    }
                }
                let frames: Vec<_> = sources
                    .iter()
                    .map(|source| (source.id, source.snapshot()))
                    .collect();
                let tempo = frames
                    .first()
                    .map(|(_, frame)| frame.tempo)
                    .unwrap_or(120.0)
                    .max(1.0);
                let max_frequency = frames
                    .iter()
                    .map(|(_, frame)| frame.max_frequency)
                    .fold(22_050.0_f32, f32::max);
                let division = GLOBAL_SCROLL_DIVISION.load(Ordering::Relaxed).clamp(0, 6) as usize;
                let seconds_per_pixel = SCROLL_BEATS_PER_PIXEL[division] * 60.0 / tempo;
                let (history_generation, column_offset) =
                    update_shared_history(&frames, seconds_per_pixel, start_new_session);
                state.column_offset = column_offset;
                if history_generation != state.history_generation
                    || state.enabled_sources != state.rendered_sources
                {
                    render_shared_history(
                        &mut state.history,
                        &shared_history_snapshot(),
                        &state.enabled_sources,
                    );
                    state.history_generation = history_generation;
                    state.rendered_sources.clone_from(&state.enabled_sources);
                    if let Some(texture) = &mut state.texture {
                        texture.set(state.history.clone(), TextureOptions::LINEAR);
                    }
                }

                egui::Frame::new()
                    .fill(Color32::from_rgb(10, 14, 23))
                    .inner_margin(0)
                    .show(ui, |ui| {
                        ui.add_space(10.0);
                        ui.horizontal(|ui| {
                            ui.add_space(14.0);
                            ui.label(
                                egui::RichText::new("FFT WINDOW")
                                    .strong()
                                    .color(Color32::from_rgb(143, 163, 190)),
                            );
                            for (index, size) in FFT_SIZES.iter().enumerate() {
                                let selected =
                                    GLOBAL_FFT_SIZE.load(Ordering::Relaxed) == index as u32;
                                if ui.selectable_label(selected, size.to_string()).clicked() {
                                    GLOBAL_FFT_SIZE.store(index as u32, Ordering::Relaxed);
                                    setter.begin_set_parameter(&params.fft_size);
                                    setter.set_parameter(&params.fft_size, index as i32);
                                    setter.end_set_parameter(&params.fft_size);
                                }
                            }
                            ui.add_space(18.0);
                            ui.label(
                                egui::RichText::new("SCROLL")
                                    .strong()
                                    .color(Color32::from_rgb(143, 163, 190)),
                            );
                            for index in SCROLL_DISPLAY_ORDER {
                                let label = SCROLL_LABELS[index];
                                let selected =
                                    GLOBAL_SCROLL_DIVISION.load(Ordering::Relaxed) == index as u32;
                                if ui.selectable_label(selected, label).clicked() {
                                    GLOBAL_SCROLL_DIVISION.store(index as u32, Ordering::Relaxed);
                                    setter.begin_set_parameter(&params.scroll_division);
                                    setter.set_parameter(&params.scroll_division, index as i32);
                                    setter.end_set_parameter(&params.scroll_division);
                                }
                            }
                        });
                        ui.add_space(8.0);

                        let content_size = ui.available_size();
                        ui.allocate_ui_with_layout(
                            content_size,
                            egui::Layout::left_to_right(egui::Align::Min),
                            |ui| {
                                if sources.len() > 1 {
                                    ui.allocate_ui_with_layout(
                                        Vec2::new(SOURCE_PANEL_WIDTH, content_size.y),
                                        egui::Layout::left_to_right(egui::Align::Min),
                                        |ui| {
                                            ui.add_space(SOURCE_PANEL_PADDING);
                                            ui.allocate_ui_with_layout(
                                                Vec2::new(
                                                    SOURCE_PANEL_WIDTH - SOURCE_PANEL_PADDING,
                                                    content_size.y,
                                                ),
                                                egui::Layout::top_down(egui::Align::Min),
                                                |ui| {
                                                    ui.label(
                                                        egui::RichText::new("SOURCES")
                                                            .strong()
                                                            .color(Color32::from_rgb(
                                                                143, 163, 190,
                                                            )),
                                                    );
                                                    ui.add_space(4.0);
                                                    egui::ScrollArea::vertical()
                                                        .id_salt("source-list")
                                                        .max_height(ui.available_height())
                                                        .show(ui, |ui| {
                                                            for source in &sources {
                                                                let generation = source
                                                                    .name_generation
                                                                    .load(Ordering::Acquire);
                                                                let source_name = state
                                                                    .source_names
                                                                    .entry(source.id)
                                                                    .or_insert_with(|| {
                                                                        source.display_name()
                                                                    });
                                                                if source_name.0 != generation {
                                                                    *source_name =
                                                                        source.display_name();
                                                                }
                                                                ui.horizontal(|ui| {
                                                                    let mut enabled = state
                                                                        .enabled_sources
                                                                        .contains(&source.id);
                                                                    if ui
                                                                        .checkbox(
                                                                            &mut enabled,
                                                                            egui::RichText::new(
                                                                                &source_name.1,
                                                                            )
                                                                            .color(source_color(
                                                                                source.id,
                                                                            )),
                                                                        )
                                                                        .changed()
                                                                    {
                                                                        if enabled {
                                                                            state
                                                                                .enabled_sources
                                                                                .insert(source.id);
                                                                        } else {
                                                                            state
                                                                                .enabled_sources
                                                                                .remove(&source.id);
                                                                        }
                                                                    }
                                                                });
                                                            }
                                                        });
                                                },
                                            );
                                        },
                                    );
                                }

                                let (scale_rect, _) = ui.allocate_exact_size(
                                    Vec2::new(FREQUENCY_SCALE_WIDTH, content_size.y),
                                    egui::Sense::hover(),
                                );
                                let chart_size = Vec2::new(ui.available_width(), content_size.y);
                                let texture = state.texture.get_or_insert_with(|| {
                                    ui.ctx().load_texture(
                                        "live-spectro-shared",
                                        state.history.clone(),
                                        TextureOptions::LINEAR,
                                    )
                                });
                                let response = ui.add(
                                    egui::Image::new((texture.id(), chart_size))
                                        .bg_fill(Color32::from_rgb(5, 8, 14))
                                        .uv(egui::Rect::from_min_max(
                                            egui::pos2(
                                                state.column_offset / HISTORY_COLUMNS as f32,
                                                0.0,
                                            ),
                                            egui::pos2(
                                                1.0 + state.column_offset / HISTORY_COLUMNS as f32,
                                                1.0,
                                            ),
                                        )),
                                );
                                paint_frequency_scale(ui, scale_rect, response.rect, max_frequency);
                            },
                        );
                    });
            },
        )
    }

    fn initialize(
        &mut self,
        _audio_io_layout: &AudioIOLayout,
        buffer_config: &BufferConfig,
        _context: &mut impl InitContext<Self>,
    ) -> bool {
        self.sample_rate = buffer_config.sample_rate;
        true
    }

    fn reset(&mut self) {
        self.samples_since_frame = 0;
        self.rolling_samples.fill(0.0);
        self.rolling_position = 0;
        self.spectrum_sequence = 0;
    }

    fn process(
        &mut self,
        buffer: &mut Buffer,
        _aux: &mut AuxiliaryBuffers,
        context: &mut impl ProcessContext<Self>,
    ) -> ProcessStatus {
        self.source
            .last_process_ms
            .store(ui_clock_ms(), Ordering::Relaxed);
        if !self.params.editor_state.is_open() && !shared_view_active() {
            return ProcessStatus::Normal;
        }

        let tempo = context.transport().tempo.unwrap_or(120.0) as f32;
        for channel_samples in buffer.iter_samples() {
            let mut mono_sample = 0.0;
            let channel_count = channel_samples.len();
            for sample in channel_samples {
                mono_sample += *sample;
            }
            mono_sample /= channel_count.max(1) as f32;

            self.rolling_samples[self.rolling_position] = mono_sample;
            self.rolling_position = (self.rolling_position + 1) % MAX_FFT_SIZE;
            self.samples_since_frame += 1;
            if self.samples_since_frame >= 1024 {
                self.samples_since_frame = 0;
                self.analyze(tempo.max(1.0));
            }
        }

        ProcessStatus::Normal
    }
}

fn render_shared_history(
    image: &mut ColorImage,
    history: &[HistoryColumn],
    enabled_sources: &HashSet<u64>,
) {
    image.pixels.fill(Color32::from_rgb(5, 8, 14));
    let first_column = HISTORY_COLUMNS.saturating_sub(history.len());
    for (history_index, column) in history.iter().enumerate() {
        let x = first_column + history_index;
        for row in 0..SPECTRUM_BINS {
            let bin = SPECTRUM_BINS - 1 - row;
            image.pixels[row * HISTORY_COLUMNS + x] = mixed_spectrum_color(
                column
                    .sources
                    .iter()
                    .filter(|source| enabled_sources.contains(&source.id))
                    .map(|source| (source_hue(source.id), source.magnitudes[bin])),
            );
        }
    }
}

fn source_hue(id: u64) -> f32 {
    if let Some([red, green, blue]) = dark2_color(id) {
        return srgb_to_oklch_hue(red, green, blue);
    }

    (id.saturating_sub(9) as f32 * 137.507_77 + 30.0) % 360.0
}

fn source_color(id: u64) -> Color32 {
    if let Some([red, green, blue]) = dark2_color(id) {
        Color32::from_rgb(red, green, blue)
    } else {
        oklch_to_color32(0.72, 0.16, source_hue(id))
    }
}

fn dark2_color(id: u64) -> Option<[u8; 3]> {
    const DARK2: [[u8; 3]; 8] = [
        [0x1B, 0x9E, 0x77],
        [0xD9, 0x5F, 0x02],
        [0x75, 0x70, 0xB3],
        [0xE7, 0x29, 0x8A],
        [0x66, 0xA6, 0x1E],
        [0xE6, 0xAB, 0x02],
        [0xA6, 0x76, 0x1D],
        [0x66, 0x66, 0x66],
    ];
    DARK2.get(id.checked_sub(1)? as usize).copied()
}

fn srgb_to_oklch_hue(red: u8, green: u8, blue: u8) -> f32 {
    let to_linear = |channel: u8| {
        let value = channel as f32 / 255.0;
        if value <= 0.040_45 {
            value / 12.92
        } else {
            ((value + 0.055) / 1.055).powf(2.4)
        }
    };
    let red = to_linear(red);
    let green = to_linear(green);
    let blue = to_linear(blue);
    let l = 0.412_221_46 * red + 0.536_332_55 * green + 0.051_445_995 * blue;
    let m = 0.211_903_5 * red + 0.680_699_5 * green + 0.107_396_96 * blue;
    let s = 0.088_302_46 * red + 0.281_718_85 * green + 0.629_978_7 * blue;
    let l_ = l.cbrt();
    let m_ = m.cbrt();
    let s_ = s.cbrt();
    let a = 1.977_998_5 * l_ - 2.428_592_2 * m_ + 0.450_593_7 * s_;
    let b = 0.025_904_037 * l_ + 0.782_771_77 * m_ - 0.808_675_77 * s_;
    b.atan2(a).to_degrees().rem_euclid(360.0)
}

fn mixed_spectrum_color(mut sources: impl Iterator<Item = (f32, f32)>) -> Color32 {
    let Some(first) = sources.next() else {
        return Color32::from_rgb(5, 8, 14);
    };
    let Some(second) = sources.next() else {
        return viridis_color(first.1);
    };
    let mut total_energy = 0.0;
    let mut hue_x = 0.0;
    let mut hue_y = 0.0;
    let mut strongest = 0.0_f32;
    for (hue, magnitude) in [first, second].into_iter().chain(sources) {
        let energy = magnitude.clamp(0.0, 1.0).powi(2);
        let radians = hue.to_radians();
        total_energy += energy;
        hue_x += energy * radians.cos();
        hue_y += energy * radians.sin();
        strongest = strongest.max(energy);
    }
    if total_energy <= 1.0e-6 {
        return Color32::from_rgb(5, 8, 14);
    }

    let hue = hue_y.atan2(hue_x).to_degrees().rem_euclid(360.0);
    let intensity = (1.0 - (-1.4 * total_energy).exp()).clamp(0.0, 1.0);
    let coherence = (hue_x.hypot(hue_y) / total_energy).clamp(0.0, 1.0);
    let dominance = (strongest / total_energy).sqrt();
    let lightness = 0.18 + 0.68 * intensity;
    let chroma = 0.06 + 0.13 * coherence * dominance;
    oklch_to_color32(lightness, chroma, hue)
}

fn viridis_color(value: f32) -> Color32 {
    const VIRIDIS: [[u8; 3]; 5] = [
        [68, 1, 84],
        [59, 82, 139],
        [33, 145, 140],
        [94, 201, 98],
        [253, 231, 37],
    ];

    let scaled = value.clamp(0.0, 1.0) * (VIRIDIS.len() - 1) as f32;
    let lower = (scaled.floor() as usize).min(VIRIDIS.len() - 2);
    let fraction = scaled - lower as f32;
    let interpolate = |channel| {
        VIRIDIS[lower][channel] as f32 * (1.0 - fraction)
            + VIRIDIS[lower + 1][channel] as f32 * fraction
    };
    Color32::from_rgb(
        interpolate(0) as u8,
        interpolate(1) as u8,
        interpolate(2) as u8,
    )
}

fn oklch_to_color32(lightness: f32, chroma: f32, hue: f32) -> Color32 {
    let radians = hue.to_radians();
    let a = chroma * radians.cos();
    let b = chroma * radians.sin();
    let l_ = lightness + 0.396_337_78 * a + 0.215_803_76 * b;
    let m_ = lightness - 0.105_561_346 * a - 0.063_854_17 * b;
    let s_ = lightness - 0.089_484_18 * a - 1.291_485_5 * b;
    let l = l_.powi(3);
    let m = m_.powi(3);
    let s = s_.powi(3);
    let linear_to_srgb = |value: f32| {
        let value = value.clamp(0.0, 1.0);
        if value <= 0.003_130_8 {
            value * 12.92
        } else {
            1.055 * value.powf(1.0 / 2.4) - 0.055
        }
    };
    let red = linear_to_srgb(4.076_741_7 * l - 3.307_711_6 * m + 0.230_969_94 * s);
    let green = linear_to_srgb(-1.268_438 * l + 2.609_757_4 * m - 0.341_319_38 * s);
    let blue = linear_to_srgb(-0.004_196_086_3 * l - 0.703_418_6 * m + 1.707_614_7 * s);
    Color32::from_rgb(
        (red * 255.0) as u8,
        (green * 255.0) as u8,
        (blue * 255.0) as u8,
    )
}

fn paint_frequency_scale(
    ui: &egui::Ui,
    scale_rect: egui::Rect,
    chart_rect: egui::Rect,
    max_frequency: f32,
) {
    const FREQUENCIES: [f32; 14] = [
        20.0, 50.0, 100.0, 200.0, 500.0, 1_000.0, 2_000.0, 5_000.0, 10_000.0, 20_000.0, 30_000.0,
        40_000.0, 60_000.0, 80_000.0,
    ];
    let painter = ui.painter();
    let log_range = (max_frequency / 20.0).ln();

    for frequency in FREQUENCIES {
        if frequency > max_frequency {
            continue;
        }
        let fraction = (frequency / 20.0).ln() / log_range;
        let y = chart_rect.bottom() - fraction * chart_rect.height();
        painter.hline(
            chart_rect.x_range(),
            y,
            egui::Stroke::new(1.0, Color32::from_white_alpha(28)),
        );
        let label = if frequency >= 1_000.0 {
            format!("{}k", frequency as u32 / 1_000)
        } else {
            frequency.to_string()
        };
        painter.text(
            egui::pos2(scale_rect.right() - 5.0, y - 2.0),
            egui::Align2::RIGHT_BOTTOM,
            label,
            egui::FontId::monospace(10.0),
            Color32::from_white_alpha(180),
        );
    }
}

impl ClapPlugin for LiveSpectroVst {
    // Kept stable so existing sessions continue to identify the plugin after the rename.
    const CLAP_ID: &'static str = "com.antoine.fft-spectrogram";
    const CLAP_DESCRIPTION: Option<&'static str> = Some("A scrolling real-time FFT spectrogram");
    const CLAP_MANUAL_URL: Option<&'static str> = None;
    const CLAP_SUPPORT_URL: Option<&'static str> = None;
    const CLAP_FEATURES: &'static [ClapFeature] = &[
        ClapFeature::AudioEffect,
        ClapFeature::Stereo,
        ClapFeature::Mono,
        ClapFeature::Analyzer,
    ];

    fn track_name_changed(&mut self, track_name: Option<&str>) {
        self.source.set_name(track_name);
    }
}

impl Vst3Plugin for LiveSpectroVst {
    // Kept stable so Ableton projects made with the prototype remain compatible.
    const VST3_CLASS_ID: [u8; 16] = *b"AntFftSpectro001";
    const VST3_SUBCATEGORIES: &'static [Vst3SubCategory] = &[
        Vst3SubCategory::Fx,
        Vst3SubCategory::Analyzer,
        Vst3SubCategory::Stereo,
    ];

    fn track_name_changed(&mut self, track_name: Option<&str>) {
        self.source.set_name(track_name);
    }
}

nice_export_clap!(LiveSpectroVst);
nice_export_vst3!(LiveSpectroVst);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shared_history_is_right_aligned() {
        let mut image = ColorImage::filled([HISTORY_COLUMNS, SPECTRUM_BINS], Color32::BLACK);
        let mut magnitudes = [0.0; SPECTRUM_BINS];
        magnitudes[0] = 1.0;
        let history = [HistoryColumn {
            sources: vec![HistorySource { id: 1, magnitudes }],
        }];

        render_shared_history(&mut image, &history, &HashSet::from([1]));

        assert_eq!(
            image.pixels[SPECTRUM_BINS * HISTORY_COLUMNS - 1],
            viridis_color(1.0)
        );
        assert_eq!(image.pixels[0], Color32::from_rgb(5, 8, 14));
    }

    #[test]
    fn shared_history_respects_source_selection() {
        let mut image = ColorImage::filled([HISTORY_COLUMNS, SPECTRUM_BINS], Color32::BLACK);
        let history = [HistoryColumn {
            sources: vec![HistorySource {
                id: 1,
                magnitudes: [1.0; SPECTRUM_BINS],
            }],
        }];

        render_shared_history(&mut image, &history, &HashSet::new());

        assert_eq!(
            image.pixels[HISTORY_COLUMNS - 1],
            Color32::from_rgb(5, 8, 14)
        );
    }

    #[test]
    fn one_source_uses_viridis() {
        assert_eq!(
            mixed_spectrum_color([(source_hue(3), 0.0)].into_iter()),
            Color32::from_rgb(68, 1, 84)
        );
        assert_eq!(
            mixed_spectrum_color([(source_hue(3), 1.0)].into_iter()),
            Color32::from_rgb(253, 231, 37)
        );
    }

    #[test]
    fn first_sources_use_dark2() {
        assert_eq!(source_color(1), Color32::from_rgb(0x1B, 0x9E, 0x77));
        assert_eq!(source_color(8), Color32::from_rgb(0x66, 0x66, 0x66));
        assert!(dark2_color(9).is_none());
    }

    #[test]
    fn mixing_equal_opposite_hues_reduces_chroma() {
        let mixed = mixed_spectrum_color([(0.0, 1.0), (180.0, 1.0)].into_iter());
        let channels = [mixed.r(), mixed.g(), mixed.b()];
        let spread = channels.iter().max().unwrap() - channels.iter().min().unwrap();

        assert!(spread < 80);
    }

    #[test]
    fn silent_spectrum_uses_chart_background() {
        assert_eq!(
            mixed_spectrum_color(std::iter::empty()),
            Color32::from_rgb(5, 8, 14)
        );
    }

    #[test]
    fn defaults_are_high_resolution_and_fast_scroll() {
        let params = LiveSpectroParams::default();
        assert_eq!(FFT_SIZES[params.fft_size.value() as usize], 8192);
        assert_eq!(
            SCROLL_LABELS[params.scroll_division.value() as usize],
            "1/32"
        );
    }

    #[test]
    fn source_name_falls_back_for_missing_or_blank_metadata() {
        let source = SharedSource::new();
        let fallback = format!("Source {}", source.id);
        assert_eq!(source.display_name().1, fallback);

        source.set_name(Some("  Drums  "));
        assert_eq!(source.display_name().1, "Drums");

        source.set_name(Some("  "));
        assert_eq!(source.display_name().1, fallback);
    }
}
