#![forbid(unsafe_code)]

use egui::{Color32, ColorImage, TextureHandle, TextureOptions, Vec2};
use nice_plug::prelude::*;
use nice_plug_egui::{EguiState, create_egui_editor};
use realfft::num_complex::Complex32;
use realfft::{RealFftPlanner, RealToComplex};
use std::{sync::Arc, time::Instant};

const FFT_SIZES: [usize; 6] = [256, 512, 1024, 2048, 4096, 8192];
const SCROLL_BEATS_PER_PIXEL: [f32; 7] = [0.25, 0.5, 1.0, 2.0, 4.0, 0.125, 0.0625];
const SCROLL_LABELS: [&str; 7] = ["1/16", "1/8", "1/4", "1/2", "1 bar", "1/32", "1/64"];
const SCROLL_DISPLAY_ORDER: [usize; 7] = [6, 5, 0, 1, 2, 3, 4];
const MAX_FFT_SIZE: usize = 8192;
const SPECTRUM_BINS: usize = 192;
const HISTORY_COLUMNS: usize = 600;
const WINDOW_WIDTH: u32 = 760;
const WINDOW_HEIGHT: u32 = 480;

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
    spectrum_tx: triple_buffer::Input<SpectrumFrame>,
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

#[derive(Clone)]
struct SpectrumFrame {
    magnitudes: [f32; SPECTRUM_BINS],
    sequence: u64,
    max_frequency: f32,
    tempo: f32,
}

struct GuiState {
    spectrum_rx: triple_buffer::Output<SpectrumFrame>,
    history: ColorImage,
    texture: Option<TextureHandle>,
    last_sequence: u64,
    column_accumulator: f32,
    last_scroll_update: Instant,
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
        let (spectrum_tx, spectrum_rx) = triple_buffer::triple_buffer(&SpectrumFrame::default());

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
            spectrum_tx,
            initial_gui_state: Some(GuiState {
                spectrum_rx,
                history: ColorImage::filled(
                    [HISTORY_COLUMNS, SPECTRUM_BINS],
                    Color32::from_rgb(5, 8, 14),
                ),
                texture: None,
                last_sequence: 0,
                column_accumulator: 0.0,
                last_scroll_update: Instant::now(),
            }),
        }
    }
}

impl LiveSpectroVst {
    fn analyze(&mut self) {
        let plan_index = self.params.fft_size.value().clamp(0, 5) as usize;
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
        let frame = self.spectrum_tx.input_buffer_mut();
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
        self.spectrum_tx.publish();
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
        let gui_state = self.initial_gui_state.take().unwrap();

        create_egui_editor(
            params.editor_state.clone(),
            gui_state,
            Default::default(),
            |_ctx, _queue, state| {
                // Texture IDs belong to one renderer and become invalid when a host closes the UI.
                state.texture = None;
                state.last_scroll_update = Instant::now();
            },
            move |ui, setter, _queue, state| {
                let frame = state.spectrum_rx.read();
                let elapsed = state.last_scroll_update.elapsed().as_secs_f32().min(0.1);
                state.last_scroll_update = Instant::now();
                let division = params.scroll_division.value().clamp(0, 6) as usize;
                let seconds_per_pixel = SCROLL_BEATS_PER_PIXEL[division] * 60.0 / frame.tempo;
                state.column_accumulator += elapsed / seconds_per_pixel;
                if frame.sequence != state.last_sequence {
                    state.last_sequence = frame.sequence;
                }
                let columns = state.column_accumulator.floor() as usize;
                state.column_accumulator -= columns as f32;
                if columns > 0 {
                    append_spectrum_columns(
                        &mut state.history,
                        &frame.magnitudes,
                        columns.min(HISTORY_COLUMNS),
                    );
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
                                let selected = params.fft_size.value() == index as i32;
                                if ui.selectable_label(selected, size.to_string()).clicked() {
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
                                let selected = params.scroll_division.value() == index as i32;
                                if ui.selectable_label(selected, label).clicked() {
                                    setter.begin_set_parameter(&params.scroll_division);
                                    setter.set_parameter(&params.scroll_division, index as i32);
                                    setter.end_set_parameter(&params.scroll_division);
                                }
                            }
                        });
                        ui.add_space(8.0);

                        let chart_size = Vec2::new(ui.available_width(), ui.available_height());
                        let texture = state.texture.get_or_insert_with(|| {
                            ui.ctx().load_texture(
                                "live-spectro",
                                state.history.clone(),
                                TextureOptions::LINEAR,
                            )
                        });
                        let response = ui.add(
                            egui::Image::new((texture.id(), chart_size))
                                .bg_fill(Color32::from_rgb(5, 8, 14))
                                .uv(egui::Rect::from_min_max(
                                    egui::pos2(
                                        state.column_accumulator / HISTORY_COLUMNS as f32,
                                        0.0,
                                    ),
                                    egui::pos2(
                                        1.0 + state.column_accumulator / HISTORY_COLUMNS as f32,
                                        1.0,
                                    ),
                                )),
                        );
                        paint_frequency_scale(ui, response.rect, frame.max_frequency);
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
        if !self.params.editor_state.is_open() {
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
                self.spectrum_tx.input_buffer_mut().tempo = tempo.max(1.0);
                self.analyze();
            }
        }

        ProcessStatus::Normal
    }
}

fn append_spectrum_columns(image: &mut ColorImage, spectrum: &[f32], columns: usize) {
    let width = image.size[0];
    for row in 0..image.size[1] {
        let row_start = row * width;
        image
            .pixels
            .copy_within(row_start + columns..row_start + width, row_start);
        let magnitude = spectrum[SPECTRUM_BINS - 1 - row];
        let color = spectrogram_color(magnitude);
        image.pixels[row_start + width - columns..row_start + width].fill(color);
    }
}

fn spectrogram_color(value: f32) -> Color32 {
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

fn paint_frequency_scale(ui: &egui::Ui, rect: egui::Rect, max_frequency: f32) {
    const FREQUENCIES: [f32; 14] = [
        20.0, 50.0, 100.0, 200.0, 500.0, 1_000.0, 2_000.0, 5_000.0, 10_000.0, 20_000.0, 30_000.0,
        40_000.0, 60_000.0, 80_000.0,
    ];
    let painter = ui.painter_at(rect);
    let log_range = (max_frequency / 20.0).ln();

    for frequency in FREQUENCIES {
        if frequency > max_frequency {
            continue;
        }
        let fraction = (frequency / 20.0).ln() / log_range;
        let y = rect.bottom() - fraction * rect.height();
        painter.hline(
            rect.x_range(),
            y,
            egui::Stroke::new(1.0, Color32::from_white_alpha(28)),
        );
        let label = if frequency >= 1_000.0 {
            format!("{}k", frequency as u32 / 1_000)
        } else {
            frequency.to_string()
        };
        painter.text(
            egui::pos2(rect.left() + 5.0, y - 2.0),
            egui::Align2::LEFT_BOTTOM,
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
}

impl Vst3Plugin for LiveSpectroVst {
    // Kept stable so Ableton projects made with the prototype remain compatible.
    const VST3_CLASS_ID: [u8; 16] = *b"AntFftSpectro001";
    const VST3_SUBCATEGORIES: &'static [Vst3SubCategory] = &[
        Vst3SubCategory::Fx,
        Vst3SubCategory::Analyzer,
        Vst3SubCategory::Stereo,
    ];
}

nice_export_clap!(LiveSpectroVst);
nice_export_vst3!(LiveSpectroVst);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn appending_spectrum_scrolls_left() {
        let mut image = ColorImage::filled([4, SPECTRUM_BINS], Color32::BLACK);
        let mut spectrum = [0.0; SPECTRUM_BINS];
        spectrum[0] = 1.0;

        append_spectrum_columns(&mut image, &spectrum, 1);

        assert_eq!(
            image.pixels[(SPECTRUM_BINS - 1) * 4 + 3],
            spectrogram_color(1.0)
        );
        assert_eq!(image.pixels[3], spectrogram_color(0.0));
    }

    #[test]
    fn viridis_uses_reference_endpoints() {
        assert_eq!(spectrogram_color(0.0), Color32::from_rgb(68, 1, 84));
        assert_eq!(spectrogram_color(1.0), Color32::from_rgb(253, 231, 37));
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
}
